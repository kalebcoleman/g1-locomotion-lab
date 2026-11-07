"""Lane B: staged, grouped-reward get-up with an assistive-pulling-force curriculum (no mocap reference).

Ideas adapted (with attribution) from
  * HoST  - Huang et al., "Learning Humanoid Standing-up Control across Diverse Postures", RSS 2025,
            arXiv:2502.08378, https://github.com/InternRobotics/HoST  (MIT licence): height-staged task rewards (right body ->
            rise -> stand, stage boundaries ~35 % / 70 % of robot height), reward groups (task / regularisation / style /
            post-task), assistive upward pulling force on the torso (~60 % of robot weight) applied while the base is low and
            decayed by a curriculum as success rises, hip-yaw/roll and shoulder-roll style penalties.
  * HumanUP - Dong et al., "Learning Getting-Up Policies for Real-World Humanoid Robots", RSS 2025, arXiv:2502.12152,
            https://github.com/RunpeiDong/HumanUP (Apache-2.0): monotonic height/upright task shaping, smoothness
            and torque regularisation, "hard-to-easy" idea (start with an easier, assisted problem).
No code is copied; the terms are re-implemented against Isaac Lab 3.0 / Newton. Differences from Lane A: see lanes/b/NOTES.md.

Assist: a world-frame +z force on `torso_link` set every control step through `asset.permanent_wrench_composer`
(applied inside Articulation.write_data_to_sim before each physics sub-step). The force scale lives in
`env._assist_scale` and DEFAULTS TO 0, so any env that does not run the curriculum term (eval_recovery.py, record_recovery.py: they
only reset/step, the curriculum manager is only invoked from reset) has exactly zero assist. The training curriculum
starts the scale at ASSIST_START and decays it to 0 as the recent global success rate rises.
"""
from __future__ import annotations
import os
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from . import common as K
from .env_cfg import G1RecoveryBaseEnvCfg
from .lane_a import promote_demote   # SAME per-env pose-level mechanism as Lane A (matched start distribution)

H1, H2 = 0.35 * K.STAND_H, 0.70 * K.STAND_H      # HoST stage boundaries (pelvis height): 0.26 m, 0.52 m
ASSIST_FRAC = 0.6        # pull force = ASSIST_FRAC * scale * robot weight  (HoST: ~60 %)
ASSIST_H = 0.70 * K.STAND_H      # assist only while pelvis height is below this (= H2 = 0.518 m)
ASSIST_START = 1.0       # initial scale (training only)
ASSIST_DECAY_PER_GEN = 0.08      # scale decrease per "generation" (= num_envs episode resets) while success is high
ASSIST_SUCC_THRESH = 0.50        # recent global rate of 'reached the standing criterion at least once' above which the assist decays (v3; v1/v2 used full 2 s success > 0.30, which never triggered)
EMA_PER_GEN = 0.5        # EMA weight of one full generation of episode results
TORSO = "torso_link"


def _robot(env):
    return env.scene["robot"]


def _h(env):
    return _robot(env).data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def _torso_id(env):
    if not hasattr(env, "_torso_bid"):
        env._torso_bid = _robot(env).find_bodies(TORSO)[0][0]
    return env._torso_bid


def torso_up(env):
    """World-up component of the torso_link z axis, in [-1, 1] (HoST/HumanUP 'head/torso upright' term)."""
    q = _robot(env).data.body_link_quat_w.torch[:, _torso_id(env)]          # (x, y, z, w)
    return 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)


def _smooth(x, lo, hi):
    t = torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------- task rewards (staged by pelvis height)
def stage_right_body(env):
    """Stage 1 (pelvis below ~H2): get the torso upright/seated; fades out as the robot rises."""
    return torch.clamp(torso_up(env), 0.0, 1.0) * (1.0 - _smooth(_h(env), H1, H2))


def stage_rise(env):
    """Stage 2: monotonic pelvis-height ramp (gradient even on the floor), scaled up by torso uprightness so a heap is not rewarded."""
    return torch.clamp(_h(env) / K.STAND_H, 0.0, 1.0) * (0.4 + 0.6 * torch.clamp(torso_up(env), 0.0, 1.0))


def stage_stand(env):
    """Stage 3 (pelvis above H2): sharp upright term, only once high enough."""
    u = torch.clamp(torso_up(env), 0.0, 1.0)
    return _smooth(_h(env), H2, K.SUCCESS_H) * torch.exp(-((1.0 - u) / 0.05))


# ----------------------------------------------------------------------------- post-task rewards (only while standing)
def post_still(env):
    r = _robot(env)
    v = torch.sum(torch.square(r.data.root_lin_vel_b.torch), -1) + 0.5 * torch.sum(torch.square(r.data.root_ang_vel_b.torch), -1)
    return K.standing_mask(env).float() * torch.exp(-2.0 * v)


def post_default_pose(env, asset_cfg: SceneEntityCfg):
    r = _robot(env)
    ids = asset_cfg.joint_ids
    dev = torch.mean(torch.abs(r.data.joint_pos.torch[:, ids] - r.data.default_joint_pos.torch[:, ids]), -1)
    return K.standing_mask(env).float() * torch.exp(-4.0 * dev)


# ----------------------------------------------------------------------------- assist force + curriculum
def assist_hook(env, env_ids=None):
    """'Termination' slot used as a per-step hook (never terminates, like track_recovery): sets the pulling force for the NEXT step.
    Exactly zero (and no wrench call at all) unless env._assist_scale > 0."""
    r = _robot(env)
    scale = getattr(env, "_assist_scale", 0.0)
    active = scale > 0.0
    if active or getattr(env, "_assist_was_on", False):
        n = env.num_envs
        f = torch.zeros(n, 1, 3, device=env.device)
        if active:
            if not hasattr(env, "_robot_weight"):
                env._robot_weight = float(r.data.body_mass.torch[0].sum()) * 9.81
            gate = 1.0 - _smooth(_h(env), ASSIST_H - 0.06, ASSIST_H)       # 1 below ~0.46 m, tapering to 0 at 0.518 m
            f[:, 0, 2] = ASSIST_FRAC * scale * env._robot_weight * gate
        env._assist_f = f[:, 0, 2]
        r.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f),
                                                                 body_ids=[_torso_id(env)], env_ids=None, is_global=True)
        env._assist_was_on = active
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def assist_curriculum(env, env_ids):
    """Global scalar curriculum (unlike Lane A's per-env pose levels): scale starts at ASSIST_START, never increases, and is
    decayed once the recent global success rate (EMA over finished episodes) exceeds ASSIST_SUCC_THRESH. Returns the scale."""
    if not hasattr(env, "_assist_scale"):
        # eval/record scripts always set cfg.fixed_poses_path (held-out poses); training never does -> assist is training-only
        eval_mode = env.cfg.fixed_poses_path is not None or os.environ.get("G1R_NO_ASSIST", "0") == "1"
        env._assist_scale = 0.0 if eval_mode else ASSIST_START; env._succ_ema = 0.0
    if env.cfg.fixed_poses_path is not None:
        env._assist_scale = 0.0
        return 0.0
    if isinstance(env_ids, slice) or len(env_ids) == 0 or env.common_step_counter == 0:   # initial full reset / nothing finished
        return env._assist_scale
    n = len(env_ids)
    gen = n / env.num_envs
    tr = K.get_tracker(env); ok = (tr.ever_stood_s if K.STRICT else tr.ever_stood)[env_ids].float().mean()
    w = min(1.0, EMA_PER_GEN * gen)
    env._succ_ema = (1 - w) * env._succ_ema + w * float(ok)
    if env._succ_ema > ASSIST_SUCC_THRESH:
        env._assist_scale = max(0.0, env._assist_scale - ASSIST_DECAY_PER_GEN * gen)
    return env._assist_scale


def success_ema(env, env_ids):
    return getattr(env, "_succ_ema", 0.0)


# ----------------------------------------------------------------------------- joint groups (hand joints are excluded: their default is outside soft limits)
BODY_JOINTS = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "torso_joint", ".*_shoulder_.*", ".*_elbow_.*"]
POSE_JOINTS = BODY_JOINTS
BODY = SceneEntityCfg("robot", joint_names=BODY_JOINTS)
POSE = SceneEntityCfg("robot", joint_names=POSE_JOINTS)


@configclass
class RewardsB:
    # --- task group (staged by pelvis height)
    right_body = RewTerm(func=stage_right_body, weight=1.0)
    rise = RewTerm(func=stage_rise, weight=3.0)
    stand = RewTerm(func=stage_stand, weight=2.0)
    # --- post-task group (only while standing)
    still = RewTerm(func=post_still, weight=2.0)
    stand_pose = RewTerm(func=post_default_pose, weight=1.5, params={"asset_cfg": POSE})
    # --- regularisation group
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": BODY})
    dof_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0, params={"asset_cfg": BODY})
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-7, params={"asset_cfg": BODY})
    # --- style group (HoST: hip yaw/roll, shoulder roll, waist (torso_joint) deviation)
    hip_yaw_roll = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])})
    shoulder_roll = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1,
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_roll_joint"])})
    waist = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])})


@configclass
class CurrB:
    levels = CurrTerm(func=promote_demote)          # per-env pose level, identical to Lane A
    assist = CurrTerm(func=assist_curriculum)       # global scalar assist scale (training only)
    succ_ema = CurrTerm(func=success_ema)


@configclass
class G1RecoveryBEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = RewardsB()
        self.curriculum = CurrB()
        self.terminations.assist = DoneTerm(func=assist_hook)
