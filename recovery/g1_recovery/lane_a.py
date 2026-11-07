"""Lane A: pure task-reward recovery with a per-environment fallen-pose curriculum (promote on success, demote on no progress)."""
from __future__ import annotations
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from . import common as K
from .env_cfg import G1RecoveryBaseEnvCfg


def upright(env):
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b.torch[:, 2], 0.0, 1.0)


def height_ramp(env):
    h = env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp(h / K.STAND_H, 0.0, 1.0)


def standing_bonus(env):
    """1 while the success criterion holds (pelvis high and torso upright), extra for being still."""
    r = env.scene["robot"]
    m = K.standing_mask(env).float()
    still = torch.exp(-torch.sum(torch.square(r.data.root_lin_vel_b.torch), -1) - 0.5 * torch.sum(torch.square(r.data.root_ang_vel_b.torch), -1))
    return m * (0.5 + 0.5 * still)


def default_pose_when_upright(env):
    r = env.scene["robot"]
    dev = torch.mean(torch.abs(r.data.joint_pos.torch - r.data.default_joint_pos.torch), -1)
    return (upright(env) > 0.9).float() * torch.exp(-3.0 * dev)


def promote_demote(env, env_ids):
    """Curriculum: success -> next level; never rose (max height < 0.35 m) -> previous level. Returns mean level."""
    t = K.get_tracker(env); lv = K.get_levels(env)
    if getattr(env, "_rec_level_fixed", False):      # harvest/eval scripts pin the level
        return lv.float().mean()
    ok = (t.succ_s if K.STRICT else t.succ)[env_ids]; bad = (~ok) & (t.max_h[env_ids] < 0.35)
    lv[env_ids] = torch.clamp(lv[env_ids] + ok.long() - bad.long(), 0, K.NUM_LEVELS - 1)
    return lv.float().mean()


import os as _os
_LAUNCHPEN = float(_os.environ.get("G1R_LAUNCHPEN", "0"))     # no-launch regulariser weight (pelvis speed excess + free flight); 0 = off
_LAUNCHTERM = float(_os.environ.get("G1R_LAUNCHTERM", "0"))   # terminate when pelvis speed exceeds this (m/s) (HumanUP-style terminate_on_velocity); 0 = off


def pelvis_speed_excess(env):
    v = torch.linalg.norm(env.scene["robot"].data.root_lin_vel_w.torch, dim=-1)
    return torch.clamp(v - 1.0, 0.0, 4.0)


def free_flight(env):
    f = torch.linalg.norm(env.scene.sensors["contact_forces"].data.net_forces_w.torch, dim=-1).sum(-1)
    return (f < 1.0).float()


def launch_terminate(env):
    return torch.linalg.norm(env.scene["robot"].data.root_lin_vel_w.torch, dim=-1) > _LAUNCHTERM


_VELPEN = float(_os.environ.get("G1R_VELPEN", "0"))     # joint velocity-limit penalty weight (0 = off; HoST/APEX-style regulariser)
_BODY = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "torso_joint", ".*_shoulder_.*", ".*_elbow_.*"])


@configclass
class RewardsA:
    upright = RewTerm(func=upright, weight=1.0)
    height = RewTerm(func=height_ramp, weight=2.0)
    stand = RewTerm(func=standing_bonus, weight=5.0)
    pose = RewTerm(func=default_pose_when_upright, weight=1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    torques = RewTerm(func=mdp.joint_torques_l2, weight=-5e-7)
    launch_speed = RewTerm(func=pelvis_speed_excess, weight=-_LAUNCHPEN) if _LAUNCHPEN > 0 else None
    flight = RewTerm(func=free_flight, weight=-_LAUNCHPEN) if _LAUNCHPEN > 0 else None
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-_VELPEN, params={"soft_ratio": 0.9, "asset_cfg": _BODY}) if _VELPEN > 0 else None


@configclass
class CurrA:
    levels = CurrTerm(func=promote_demote)


@configclass
class G1RecoveryAEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards = RewardsA()
        self.curriculum = CurrA()
        if _LAUNCHTERM > 0:
            from isaaclab.managers import TerminationTermCfg as _DoneTerm
            self.terminations.launch = _DoneTerm(func=launch_terminate)


@configclass
class G1RecoveryAAssistEnvCfg(G1RecoveryAEnvCfg):
    """Lane A + HoST-style gated per-env assist (active only when G1R_ASSIST_TRAIN=1 / G1R_EVAL_ASSIST is set; see assist.py)."""
    def __post_init__(self):
        super().__post_init__()
        from .assist import add_assist
        add_assist(self)


# ---------------------------------------------------------------------------------------------------------- baked recipes (no env vars; usable from `isaaclab__train`)
@configclass
class RewardsR1(RewardsA):
    """R1 = the V11 recipe: Lane A task rewards + joint-velocity-limit penalty (10) + no-launch penalties (pelvis speed above 1 m/s, free flight; 5 each)."""
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-10.0, params={"soft_ratio": 0.9, "asset_cfg": _BODY})
    launch_speed = RewTerm(func=pelvis_speed_excess, weight=-5.0)
    flight = RewTerm(func=free_flight, weight=-5.0)


@configclass
class RewardsR2(RewardsR1):
    """R2 = R1 + a strong joint-POSITION-limit penalty on body joints only (the V11/V12 policies sat at position limits on 40-50% of steps; hands excluded:
    their default pose is outside their soft limits, so the old all-joints term was a constant offset)."""
    dof_limits = None
    dof_limits_body = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": _BODY})


def _recipe(cfg, rewards, assist_train, start_level):
    from .assist import add_assist
    cfg.rewards = rewards(); cfg.curriculum = CurrA(); add_assist(cfg)
    cfg.assist_train = assist_train; cfg.start_level = start_level
    return cfg


@configclass
class G1RecoveryR1EnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR1, True, 1)


@configclass
class G1RecoveryR1EvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR1, False, 0)


@configclass
class G1RecoveryR2EnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR2, True, 1)


@configclass
class G1RecoveryR2EvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR2, False, 0)


@configclass
class G1RecoveryR3EnvCfg(G1RecoveryBaseEnvCfg):
    """R3 = R2 + assist decay only after a supported stand held >= 1 s, 0.1 per such episode (R1/R2 lost the assist after ~100 iterations because
    the legacy criterion, which the assist itself satisfies, decayed it by 0.2 per episode)."""
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR2, True, 1); self.assist_decay_on = "supported1s"; self.assist_decay = 0.1


@configclass
class G1RecoveryR3EvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR2, False, 0)


# ---------------------------------------------------------------------------------------------------------- v13: lying-dominant fixed start mix
POSE_MIX = [0.05, 0.10, 0.15, 0.30, 0.40]     # P(level): stand, sit bank, mid-rise bank, lying 70-90 deg (noise 0.3), lying 80-90 deg (noise 0.4 = held-out-like)


@configclass
class CurrMix:
    lying_ever_supported = CurrTerm(func=K.mix_stat, params={"levels": [3, 4], "which": "ever_stood_s"})
    lying_strict_hold = CurrTerm(func=K.mix_stat, params={"levels": [3, 4], "which": "succ_s"})
    lying_max_h = CurrTerm(func=K.mix_stat, params={"levels": [3, 4], "which": "max_h"})
    sit_ever_supported = CurrTerm(func=K.mix_stat, params={"levels": [1, 2], "which": "ever_stood_s"})
    stand_strict_hold = CurrTerm(func=K.mix_stat, params={"levels": [0], "which": "succ_s"})


def _recipe_mix(cfg, assist, scale=None):
    """R2 rewards + fixed lying-dominant start mix instead of the promote/demote levels (which stayed at the stand/sit levels in every non-exploit run)."""
    cfg.rewards = RewardsR2(); cfg.curriculum = CurrMix(); cfg.pose_mix = POSE_MIX
    if assist:
        from .assist import add_assist
        add_assist(cfg); cfg.assist_train = True; cfg.assist_decay_on = "supported1s"; cfg.assist_decay = 0.1
    if scale is not None:
        cfg.actions.joint_pos.scale = scale
    return cfg


@configclass
class G1RecoveryR4EnvCfg(G1RecoveryBaseEnvCfg):
    """R4 = R3 with the fixed lying-dominant start mix (single change vs R3)."""
    def __post_init__(self):
        super().__post_init__(); _recipe_mix(self, True)


@configclass
class G1RecoveryR4nEnvCfg(G1RecoveryBaseEnvCfg):
    """R4n = R4 without any assist (single change vs R4)."""
    def __post_init__(self):
        super().__post_init__(); _recipe_mix(self, False)


@configclass
class G1RecoveryR5EnvCfg(G1RecoveryBaseEnvCfg):
    """R5 = R4 with joint-position action scale 1.0 instead of 0.5 (single change vs R4). Warm start uses an exactly converted checkpoint (tools/rescale_ckpt.py)."""
    def __post_init__(self):
        super().__post_init__(); _recipe_mix(self, True, scale=1.0)


@configclass
class G1RecoveryR5EvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _recipe(self, RewardsR2, False, 0); self.actions.joint_pos.scale = 1.0


# ---------------------------------------------------------------------------------------------------------- v14: height-dominant reward (idea from g1-humanoid-rl, MIT, ch. 14: monotonic floor-to-stand ramp weight 10, upright 2)
def height_ramp_floor(env, floor=0.1):
    h = env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return torch.clamp((h - floor) / (K.STAND_H - floor), 0.0, 1.0)


@configclass
class RewardsR6(RewardsR2):
    """R2 with a height-dominant task reward: monotonic ramp from the floor (0.1 m) to the standing height, weight 10 (R2: 2); upright weight 2 (R2: 1)."""
    height = RewTerm(func=height_ramp_floor, weight=10.0)
    upright = RewTerm(func=upright, weight=2.0)


POSE_MIX_LYING = [0.0, 0.0, 0.0, 0.5, 0.5]


@configclass
class G1RecoveryR6bEnvCfg(G1RecoveryBaseEnvCfg):
    """R6b = R4n with the height-dominant reward (single change vs R4n)."""
    def __post_init__(self):
        super().__post_init__(); _recipe_mix(self, False); self.rewards = RewardsR6()


@configclass
class G1RecoveryR6cEnvCfg(G1RecoveryBaseEnvCfg):
    """R6c = R6b with lying-only starts (single change vs R6b); meant to be trained from scratch."""
    def __post_init__(self):
        super().__post_init__(); _recipe_mix(self, False); self.rewards = RewardsR6(); self.pose_mix = POSE_MIX_LYING


# ---------------------------------------------------------------------------------------------------------- v15: fix the forward-fold optimum seen at the R6c check
def upright_full(env):
    """(1 + up) / 2 in [0, 1]: unlike `upright` (clamped at 0), it has a gradient when the torso is upside down (R6 policies hold up = -0.97)."""
    return 0.5 * (1.0 - env.scene["robot"].data.projected_gravity_b.torch[:, 2])


def imu_height_ramp(env, floor=0.1, target=0.85):
    """Upper-body height: imu_link (0.85 m standing, 0.38 m in the fold). head_link/torso_link report the pelvis origin in this USD, so they cannot be used."""
    r = env.scene["robot"]
    if not hasattr(env, "_imu_bid"):
        env._imu_bid = r.find_bodies("imu_link")[0][0]
    h = r.data.body_pos_w.torch[:, env._imu_bid, 2] - env.scene.env_origins[:, 2]
    return torch.clamp((h - floor) / (target - floor), 0.0, 1.0)


def height_ramp_gated(env):
    """Pelvis ramp x (0.25 + 0.75 * upright_full): a folded robot earns only a quarter of the height reward (HoST: orientation x height)."""
    return height_ramp_floor(env) * (0.25 + 0.75 * upright_full(env))


@configclass
class RewardsR7u(RewardsR6):
    """R6 with `upright_full` (single change: orientation gradient below horizontal)."""
    upright = RewTerm(func=upright_full, weight=2.0)


@configclass
class RewardsR7h(RewardsR6):
    """R6 with the height term on imu_link instead of the pelvis (single change)."""
    height = RewTerm(func=imu_height_ramp, weight=10.0)


@configclass
class RewardsR7m(RewardsR6):
    """R6 with the pelvis height gated by torso orientation (single change)."""
    height = RewTerm(func=height_ramp_gated, weight=10.0)


def _r7(cfg, rewards):
    _recipe_mix(cfg, False); cfg.rewards = rewards(); cfg.pose_mix = POSE_MIX_LYING
    return cfg


@configclass
class G1RecoveryR7uEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _r7(self, RewardsR7u)


@configclass
class G1RecoveryR7hEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _r7(self, RewardsR7h)


@configclass
class G1RecoveryR7mEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _r7(self, RewardsR7m)


# --- v16 (R8 family). Base R8 = R7m with the stand bonus on the feet-supported mask; E1 = R8f; E2 = R8s / R8p; see EXPERIMENT_LOG.md
def standing_bonus_supported(env):
    """standing_bonus on the same feet-supported condition the evaluator requires (the R1-R7 recipes used the pelvis + tilt mask, so unsupported stands were rewarded)."""
    r = env.scene["robot"]
    m = (K.standing_mask(env) & K.supported_mask(env)).float()
    still = torch.exp(-torch.sum(torch.square(r.data.root_lin_vel_b.torch), -1) - 0.5 * torch.sum(torch.square(r.data.root_ang_vel_b.torch), -1))
    return m * (0.5 + 0.5 * still)


_PROP_WORDS = {"palm", "zero", "one", "two", "three", "four", "five", "six", "elbow", "knee", "head"}


def _contact_groups(env):
    if not hasattr(env, "_cg_feet"):
        names = list(env.scene.sensors["contact_forces"].body_names)
        env._cg_feet = [i for i, n in enumerate(names) if "ankle_roll" in n]
        env._cg_prop = [i for i, n in enumerate(names) if _PROP_WORDS & set(n.split("_"))]
    return env._cg_feet, env._cg_prop


def feet_load_share(env):
    """Share of all contact force carried by the feet, paid only while both feet are loaded."""
    feet, _ = _contact_groups(env)
    F = torch.linalg.norm(env.scene.sensors["contact_forces"].data.net_forces_w.torch, dim=-1)
    both = (F[:, feet] > K.SUPPORT_FORCE_N).all(-1).float()
    return both * F[:, feet].sum(-1) / (F.sum(-1) + 1.0)


def prop_force(env, pelvis_h=0.40, full_n=100.0):
    """Contact force on hands, elbows, knees and head once the pelvis is above `pelvis_h` (early propping from lying stays free); saturates at `full_n`."""
    _, prop = _contact_groups(env)
    F = torch.linalg.norm(env.scene.sensors["contact_forces"].data.net_forces_w.torch, dim=-1)
    h = env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (h > pelvis_h).float() * torch.clamp(F[:, prop].sum(-1) / full_n, max=1.0)


@configclass
class RewardsR8(RewardsR7m):
    """R7m with the stand bonus on the feet-supported mask (single change vs R7m)."""
    stand = RewTerm(func=standing_bonus_supported, weight=5.0)


@configclass
class RewardsR8f(RewardsR8):
    """E1: R8 + feet-load-share reward + gated prop-force penalty."""
    feet_share = RewTerm(func=feet_load_share, weight=3.0)
    prop = RewTerm(func=prop_force, weight=-2.0)


def stance_excess(env, free=0.45, full=0.90, pelvis_h=0.40):
    """Distance between the feet above `free` m, scaled to 1 at `full` m, once the pelvis is above `pelvis_h` (lying and early rising stay free)."""
    r = env.scene["robot"]
    if not hasattr(env, "_st_feet"):
        env._st_feet = [r.body_names.index("left_ankle_roll_link"), r.body_names.index("right_ankle_roll_link")]
    p = r.data.body_pos_w.torch[:, env._st_feet, :2]; d = torch.linalg.norm(p[:, 0] - p[:, 1], dim=-1)
    h = r.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return (h > pelvis_h).float() * torch.clamp((d - free) / (full - free), 0.0, 1.0)


@configclass
class RewardsR8w(RewardsR8):
    """E5: R8 + stance-width penalty (single change vs R8)."""
    stance = RewTerm(func=stance_excess, weight=-5.0)


class SettleJointPositionAction(JointPositionAction):
    """For the first `settle_steps` control steps of an episode the target follows the measured joint position: stiffness does no work and only joint damping acts, so the robot settles passively under gravity."""

    def process_actions(self, actions):
        super().process_actions(actions)
        m = self._env.episode_length_buf < self.cfg.settle_steps
        if m.any():
            self._processed_actions[m] = self._asset.data.joint_pos.torch[m][:, self._joint_ids]


@configclass
class SettleJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SettleJointPositionAction
    settle_steps: int = 30


def _r8(cfg, rewards, phi_deg=None, settle=0):
    _r7(cfg, rewards)
    if phi_deg is not None:
        cfg.pose_phi_deg = phi_deg
    if settle:
        a = cfg.actions.joint_pos
        cfg.actions.joint_pos = SettleJointPositionActionCfg(asset_name=a.asset_name, joint_names=a.joint_names, scale=a.scale, use_default_offset=a.use_default_offset, preserve_order=a.preserve_order, clip=a.clip, settle_steps=settle)
    return cfg


@configclass
class G1RecoveryR8EnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _r8(self, RewardsR8)


@configclass
class G1RecoveryR8fEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _r8(self, RewardsR8f)


@configclass
class G1RecoveryR8sEnvCfg(G1RecoveryBaseEnvCfg):
    """E2: R8, supine family only (phi 270 +- 30 deg), 30-step passive settle."""
    def __post_init__(self):
        super().__post_init__(); _r8(self, RewardsR8, phi_deg=(270.0, 30.0), settle=30)


@configclass
class G1RecoveryR8pEnvCfg(G1RecoveryBaseEnvCfg):
    """E2: R8, prone family only (phi 90 +- 30 deg), 30-step passive settle."""
    def __post_init__(self):
        super().__post_init__(); _r8(self, RewardsR8, phi_deg=(90.0, 30.0), settle=30)


@configclass
class G1RecoveryR8wEnvCfg(G1RecoveryBaseEnvCfg):
    """E5: R8 + stance-width penalty."""
    def __post_init__(self):
        super().__post_init__(); _r8(self, RewardsR8w)
