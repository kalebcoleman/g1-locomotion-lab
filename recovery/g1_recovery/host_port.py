"""Port of HoST (Huang et al., "Learning Humanoid Standing-up Control across Diverse Postures", RSS 2025, arXiv:2502.08378;
https://github.com/InternRobotics/HoST, MIT licence, commit 70bb580) to Isaac Lab 3.0.0 + Newton/MJWarp for our 37-joint G1.

Ported from `legged_gym/envs/base/host_ground.py` and `legged_gym/envs/g1/g1_config_ground(_prone).py`:
  * reward groups task / regu / style / target with HoST's term definitions and scales (task = orientation x head height, not dt-scaled);
  * actions are joint-position offsets from the CURRENT joint position, recomputed every physics substep (so the PD acts like kp*scale*a - kd*v),
    with HoST's per-env action-rescale curriculum (1.0 -> 0.25) and 30 unactuated control steps after reset;
  * HoST's vertical pulling force (200 N on the torso while it is upright-ish) with its decay curriculum, TRAINING ONLY;
  * HoST's PD gains, observation terms and scales, noise, 6-step observation history; terminations on base speed > 20 m/s or joint speed > 300 rad/s.
The multi-critic PPO is in `scripts/train_host.py`.

Deliberate differences (robot and protocol, documented in EXPERIMENT_LOG.md):
  * 37 joints (23 body + 14 finger) instead of 23; finger joints are excluded from the style/limit terms and keep our gains;
  * Unitree's real torque limits and self-collision on (HoST's URDF also enables self-collision);
  * our start-pose sampler (the DEV-like lying distribution, restricted to one fall family) instead of HoST's fixed init quaternion;
  * HoST's keyframe links do not exist in our USD: head point = torso_link + R * (0, 0, 0.45) as in HoST's URDF; foot corners use HoST's
    auxiliary-link offsets in the ankle_roll frame; target base height 0.72 m (HoST 0.75 m for its URDF; our model stands with the pelvis at ~0.70 m).
"""
from __future__ import annotations
import math
import os
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from . import common as K
from .env_cfg import G1RecoveryBaseEnvCfg

UNACTUATED = 30                 # control steps (HoST unactuated_timesteps at 50 Hz)
FORCE0 = 200.0                  # N (HoST: 100 N on torso_link + 100 N on keyframe_torso_link)
GROUP_WEIGHTS = (2.5, 0.1, 1.0, 1.0)   # task, regu, style, target (HoST reward_group_weights)
GROUPS = ("task", "regu", "style", "target")
BASE_H_PHASE1, BASE_H_PHASE3, BASE_H_TARGET = 0.45, 0.65, 0.72
HEAD_OFFSET = (0.0, 0.0, 0.45)
AUX = ((0.035, 0.1, 0.0), (0.035, -0.1, 0.0), (0.2, 0.0, 0.0), (-0.15, 0.0, 0.0))
TARGET_UPPER = {"left_shoulder_roll_joint": 0.3, "right_shoulder_roll_joint": -0.3}   # HoST target_joint_angles; other arm and waist joints 0
HOST_GAINS = {"hip": (150.0, 4.0), "knee": (200.0, 6.0), "ankle": (40.0, 2.0), "torso": (100.0, 4.0), "shoulder": (100.0, 4.0), "elbow": (100.0, 4.0)}


# ---------------------------------------------------------------- helpers
def tolerance(x, lower, upper, margin, value_at_margin):
    """dm_control-style tolerance with a Gaussian sigmoid (HoST g1_utils.tolerance)."""
    inb = (x >= lower) & (x <= upper)
    if margin == 0:
        return inb.float()
    d = torch.where(x < lower, lower - x, x - upper) / margin
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    return torch.where(inb, torch.ones_like(x), torch.exp(-0.5 * (d * scale) ** 2))


def qrot(q, v):
    """Rotate v by quaternion q (x, y, z, w)."""
    u = q[..., :3]; w = q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def _H(env):
    """Cached indices and per-env state for the HoST terms."""
    if not hasattr(env, "_host"):
        r = env.scene["robot"]; jn = list(r.joint_names); bn = list(r.body_names); dev = env.device
        J = lambda *keys: torch.tensor([i for i, n in enumerate(jn) if any(k in n for k in keys) and not K.is_hand_joint(n)], device=dev, dtype=torch.long)
        h = dict(torso=bn.index("torso_link"), feet=[bn.index("left_ankle_roll_link"), bn.index("right_ankle_roll_link")],
                 knees=[bn.index("left_knee_link"), bn.index("right_knee_link")],
                 waist=J("torso_joint"), hip_yaw=J("hip_yaw"), hip_roll=J("hip_roll"), knee=J("knee_joint"),
                 sh_roll=torch.tensor([jn.index("left_shoulder_roll_joint"), jn.index("right_shoulder_roll_joint")], device=dev),
                 body=torch.tensor([not K.is_hand_joint(n) for n in jn], device=dev),
                 upper=J("shoulder", "elbow", "torso_joint"))
        tgt = torch.zeros(len(jn), device=dev)
        for k, v in TARGET_UPPER.items():
            tgt[jn.index(k)] = v
        h["target_q"] = tgt
        h["head_off"] = torch.tensor(HEAD_OFFSET, device=dev)
        h["aux"] = torch.tensor(AUX, device=dev)
        h["force"] = torch.full((env.num_envs,), FORCE0 if getattr(env.cfg, "host_force", True) else 0.0, device=dev)
        h["rescale"] = torch.ones(env.num_envs, device=dev)
        h["last_head"] = torch.zeros(env.num_envs, device=dev)
        env._host = h
    return env._host


def _root_h(env):
    return env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]


def head_height(env):
    """HoST head height: keyframe_head (torso_link + R*(0,0,0.45)) minus the mean foot height."""
    r = env.scene["robot"]; h = _H(env)
    p = r.data.body_pos_w.torch[:, h["torso"]]; q = r.data.body_quat_w.torch[:, h["torso"]]
    head = p + qrot(q, h["head_off"].expand_as(p))
    feet = r.data.body_pos_w.torch[:, h["feet"], 2].mean(-1)
    return head[:, 2] - feet


def unactuated(env):
    return env.episode_length_buf < UNACTUATED


# ---------------------------------------------------------------- task (multiplied, not dt-scaled: weight = 1/dt cancels Isaac Lab's dt)
def task_reward(env):
    up = -env.scene["robot"].data.projected_gravity_b.torch[:, 2]
    hh = head_height(env)
    _H(env)["last_head"] = hh.detach()
    return tolerance(up, 0.99, math.inf, 1.0, 0.05) * tolerance(hh, 1.0, math.inf, 1.0, 0.1)


# ---------------------------------------------------------------- regularisation
def smoothness(env):
    t = env.action_manager.get_term("joint_pos")
    return torch.sum(torch.square(t.raw_actions - 2 * t._prev + t._prev2), -1)


def joint_power(env):
    r = env.scene["robot"]
    return torch.sum(torch.abs(r.data.joint_vel.torch) * torch.abs(r.data.applied_torque.torch), -1)


def joint_tracking_error(env):
    t = env.action_manager.get_term("joint_pos")
    return torch.sum(torch.square(t.processed_actions - env.scene["robot"].data.joint_pos.torch), -1)


def dof_pos_limits_body(env):
    r = env.scene["robot"]; h = _H(env); q = r.data.joint_pos.torch; lim = r.data.soft_joint_pos_limits.torch
    out = (-(q - lim[..., 0]).clamp(max=0.0) + (q - lim[..., 1]).clamp(min=0.0)) * h["body"]
    return out.sum(-1)


def dof_vel_limits_body(env, soft=0.9):
    r = env.scene["robot"]; h = _H(env)
    return (((r.data.joint_vel.torch.abs() - r.data.joint_vel_limits.torch * soft).clamp(0.0, 1.0)) * h["body"]).sum(-1)


# ---------------------------------------------------------------- style
def _q(env):
    return env.scene["robot"].data.joint_pos.torch


def waist_deviation(env):
    return (_q(env)[:, _H(env)["waist"]].abs() > 1.4).any(-1).float()


def _dev2(env, key):
    a = _q(env)[:, _H(env)[key]].abs()
    return ((a.max(-1).values > 1.4) | (a.min(-1).values > 0.9)).float()


def hip_yaw_deviation(env):
    return _dev2(env, "hip_yaw")


def hip_roll_deviation(env):
    return _dev2(env, "hip_roll")


def shoulder_roll_deviation(env):
    q = _q(env); i = _H(env)["sh_roll"]
    return ((q[:, i[0]] < -0.02) | (q[:, i[1]] > 0.02)).float()


def _foot_disp(env, k):
    r = env.scene["robot"]; h = _H(env)
    base = r.data.root_pos_w.torch[:, :2]; f = r.data.body_pos_w.torch[:, h["feet"][k]]
    mse = torch.sum(torch.square(base - f[:, :2]), -1).clamp(min=0.3)
    fz = f[:, 2] - env.scene.env_origins[:, 2]
    return torch.exp(-2.0 * mse) * (fz < 0.3).float() * (_root_h(env) > BASE_H_PHASE3).float()


def left_foot_displacement(env):
    return _foot_disp(env, 0)


def right_foot_displacement(env):
    return _foot_disp(env, 1)


def knee_deviation(env):
    a = _q(env)[:, _H(env)["knee"]]
    return ((a.abs().max(-1).values > 2.85) | (a.min(-1).values < -0.06)).float()


def shank_orientation(env):
    r = env.scene["robot"]; h = _H(env); bp = r.data.body_pos_w.torch
    v = bp[:, h["knees"]] - bp[:, h["feet"]]
    o = (v[..., 2] / torch.linalg.norm(v, dim=-1).clamp(min=1e-6)).mean(-1)
    return tolerance(o, 0.8, math.inf, 1.0, 0.1) * (_root_h(env) > BASE_H_PHASE1).float()


def ground_parallel(env):
    """HoST: variance of the heights of the ankle_roll link and its four auxiliary corner points (x10) below 0.05, per foot, averaged."""
    r = env.scene["robot"]; h = _H(env); var = []
    for k in range(2):
        p = r.data.body_pos_w.torch[:, h["feet"][k]]; q = r.data.body_quat_w.torch[:, h["feet"][k]]
        pts = torch.cat([p[:, None], p[:, None] + qrot(q[:, None].expand(-1, 4, -1), h["aux"].expand(p.shape[0], -1, -1))], 1)
        var.append((pts[..., 2] * 10.0).var(1))
    return (torch.stack(var, -1).mean(-1) < 0.05).float()


def feet_distance(env):
    r = env.scene["robot"]; h = _H(env); p = r.data.body_pos_w.torch[:, h["feet"]]
    return (torch.linalg.norm(p[:, 0] - p[:, 1], dim=-1) > 0.9).float()


def style_ang_vel_xy(env):
    w = env.scene["robot"].data.root_ang_vel_b.torch[:, :2]
    return torch.exp(-2.0 * torch.sum(torch.square(w), -1)) * (_root_h(env) > BASE_H_PHASE1).float()


# ---------------------------------------------------------------- post-task (target) terms, active above 0.65 m
def _up3(env):
    return (_root_h(env) > BASE_H_PHASE3).float()


def target_ang_vel_xy(env):
    w = env.scene["robot"].data.root_ang_vel_b.torch[:, :2]
    return torch.exp(-2.0 * torch.sum(torch.square(w), -1)) * _up3(env)


def target_lin_vel_xy(env):
    v = env.scene["robot"].data.root_lin_vel_b.torch[:, :2]
    return torch.exp(-5.0 * torch.sum(torch.square(v), -1)) * _up3(env)


def target_feet_height_var(env):
    r = env.scene["robot"]; h = _H(env); z = r.data.body_pos_w.torch[:, h["feet"], 2] * 10.0
    return torch.exp(-2.0 * (z[:, 0] - z[:, 1]).abs().clamp(min=0.2)) * _up3(env)


def target_upper_dof_pos(env):
    h = _H(env); q = _q(env)[:, h["upper"]]
    return torch.exp(-0.1 * torch.sum(torch.square(q - h["target_q"][h["upper"]]), -1)) * _up3(env)


def target_orientation(env):
    g = env.scene["robot"].data.projected_gravity_b.torch[:, :2]
    return torch.exp(-5.0 * torch.sum(torch.square(g), -1)) * _up3(env)


def target_base_height(env):
    return torch.exp(-20.0 * (_root_h(env) - BASE_H_TARGET).abs()) * _up3(env)


# ---------------------------------------------------------------- observations (HoST compute_observations), zeroed while unactuated
def host_obs(env):
    r = env.scene["robot"]; h = _H(env); t = env.action_manager.get_term("joint_pos")
    o = torch.cat([r.data.root_ang_vel_b.torch * 0.25, r.data.projected_gravity_b.torch, r.data.joint_pos.torch, r.data.joint_vel.torch * 0.05,
                   t.raw_actions, (h["rescale"] + (torch.rand_like(h["rescale"]) - 0.5) * 0.05)[:, None]], -1)
    if getattr(env.cfg, "host_noise", True):
        nj = r.data.joint_pos.torch.shape[1]
        sc = torch.cat([torch.full((3,), 0.2 * 0.25), torch.full((3,), 0.05), torch.full((nj,), 0.01), torch.full((nj,), 1.5 * 0.05), torch.zeros(nj + 1)]).to(env.device)
        o = o + (2 * torch.rand_like(o) - 1) * sc
    return o * (~unactuated(env)).float()[:, None]


# ---------------------------------------------------------------- terminations, curriculum, force
def base_vel_out(env, limit=20.0):
    return (torch.linalg.norm(env.scene["robot"].data.root_lin_vel_b.torch, dim=-1) > limit) & ~unactuated(env)


def dof_vel_out(env, limit=300.0):
    return (env.scene["robot"].data.joint_vel.torch.abs().max(-1).values > limit) & ~unactuated(env)


def force_curriculum(env, env_ids):
    """HoST update_force_curriculum: if the resetting envs' mean head height exceeded 0.9 m, reduce their force by 20 N and action rescale by 0.02."""
    h = _H(env)
    if isinstance(env_ids, slice) or env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if isinstance(env_ids, slice) else slice(None)]
    if len(env_ids) == 0:
        return torch.tensor(float(h["force"].mean()))
    if float(h["last_head"][env_ids].mean()) > 0.9:
        h["force"][env_ids] = (h["force"][env_ids] - 20.0).clamp(min=0.0)
        h["rescale"][env_ids] = (h["rescale"][env_ids] - 0.02).clamp(min=0.25)
    return torch.tensor(float(h["force"].mean()))


def rescale_stat(env, env_ids):
    return torch.tensor(float(_H(env)["rescale"].mean()))


class HostAction(JointPositionAction):
    """target = current q + rescale * a, recomputed at every physics substep; zero action for the first UNACTUATED control steps;
    applies the HoST pulling force (training only) at every substep."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros_like(self._raw_actions); self._prev2 = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions):
        self._prev2[:] = self._prev; self._prev[:] = self._raw_actions
        self._raw_actions[:] = actions
        self._raw_actions[unactuated(self._env)] = 0.0

    def apply_actions(self):
        h = _H(self._env); q = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._processed_actions = q + h["rescale"][:, None] * self._raw_actions
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)
        if getattr(self._env.cfg, "host_force", True):
            upright = -self._asset.data.projected_gravity_b.torch[:, 2] > 0.8
            fz = h["force"] * (upright & ~unactuated(self._env)).float()
            f = torch.zeros(self._env.num_envs, 1, 3, device=self._env.device); f[:, 0, 2] = fz
            self._asset.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), body_ids=[h["torso"]], env_ids=None, is_global=True)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._prev[:] = 0.0; self._prev2[:] = 0.0
        else:
            self._prev[env_ids] = 0.0; self._prev2[env_ids] = 0.0


@configclass
class HostActionCfg(JointPositionActionCfg):
    class_type: type = HostAction


# ---------------------------------------------------------------- managers
_DT = 0.02


@configclass
class HostRewards:
    task_task = RewTerm(func=task_reward, weight=1.0 / _DT)
    regu_dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    regu_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    regu_smoothness = RewTerm(func=smoothness, weight=-0.01)
    regu_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-6)
    regu_joint_power = RewTerm(func=joint_power, weight=-2.5e-5)
    regu_dof_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    regu_joint_tracking_error = RewTerm(func=joint_tracking_error, weight=-0.00025)
    regu_dof_pos_limits = RewTerm(func=dof_pos_limits_body, weight=-100.0)
    regu_dof_vel_limits = RewTerm(func=dof_vel_limits_body, weight=-1.0)
    style_waist_deviation = RewTerm(func=waist_deviation, weight=-10.0)
    style_hip_yaw_deviation = RewTerm(func=hip_yaw_deviation, weight=-10.0)
    style_hip_roll_deviation = RewTerm(func=hip_roll_deviation, weight=-10.0)
    style_shoulder_roll_deviation = RewTerm(func=shoulder_roll_deviation, weight=-2.5)
    style_left_foot_displacement = RewTerm(func=left_foot_displacement, weight=2.5)
    style_right_foot_displacement = RewTerm(func=right_foot_displacement, weight=2.5)
    style_knee_deviation = RewTerm(func=knee_deviation, weight=-0.25)
    style_shank_orientation = RewTerm(func=shank_orientation, weight=10.0)
    style_ground_parallel = RewTerm(func=ground_parallel, weight=20.0)
    style_feet_distance = RewTerm(func=feet_distance, weight=-10.0)
    style_style_ang_vel_xy = RewTerm(func=style_ang_vel_xy, weight=1.0)
    target_ang_vel_xy = RewTerm(func=target_ang_vel_xy, weight=10.0)
    target_lin_vel_xy = RewTerm(func=target_lin_vel_xy, weight=10.0)
    target_feet_height_var = RewTerm(func=target_feet_height_var, weight=2.5)
    target_target_upper_dof_pos = RewTerm(func=target_upper_dof_pos, weight=10.0)
    target_target_orientation = RewTerm(func=target_orientation, weight=10.0)
    target_target_base_height = RewTerm(func=target_base_height, weight=10.0)


@configclass
class HostObs:
    @configclass
    class PolicyCfg(ObsGroup):
        host = ObsTerm(func=host_obs)

        def __post_init__(self):
            self.enable_corruption = False; self.concatenate_terms = True; self.history_length = 6; self.flatten_history_dim = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class HostCurr:
    force = CurrTerm(func=force_curriculum)
    rescale = CurrTerm(func=rescale_stat)


def _host_gains(cfg):
    import copy
    act = copy.deepcopy(cfg.scene.robot.actuators)
    for a in act.values():
        st = {}; dm = {}
        for expr in a.joint_names_expr:
            for key, (kp, kd) in HOST_GAINS.items():
                if key in expr:
                    st[expr] = kp; dm[expr] = kd
        if st:
            old_s = a.stiffness if isinstance(a.stiffness, dict) else {e: a.stiffness for e in a.joint_names_expr}
            old_d = a.damping if isinstance(a.damping, dict) else {e: a.damping for e in a.joint_names_expr}
            a.stiffness = {**{e: old_s.get(e, a.stiffness if not isinstance(a.stiffness, dict) else 40.0) for e in a.joint_names_expr}, **st}
            a.damping = {**{e: old_d.get(e, a.damping if not isinstance(a.damping, dict) else 10.0) for e in a.joint_names_expr}, **dm}
    cfg.scene.robot.actuators = act


def _host(cfg, phi_deg, eval_mode=False):
    from .lane_a import real_torque_limits, self_collisions_on, add_ground_sensor, POSE_MIX_LYING
    cfg.rewards = HostRewards(); cfg.observations = HostObs(); cfg.curriculum = HostCurr(); cfg.pose_mix = POSE_MIX_LYING; cfg.pose_phi_deg = phi_deg
    a = cfg.actions.joint_pos
    cfg.actions.joint_pos = HostActionCfg(asset_name=a.asset_name, joint_names=a.joint_names, scale=1.0, use_default_offset=False, preserve_order=a.preserve_order)
    cfg.terminations.base_vel_out = DoneTerm(func=base_vel_out); cfg.terminations.dof_vel_out = DoneTerm(func=dof_vel_out)
    real_torque_limits(cfg); self_collisions_on(cfg); add_ground_sensor(cfg); _host_gains(cfg)
    cfg.host_force = not eval_mode and os.environ.get("G1R_HOST_FORCE", "1") == "1"
    cfg.host_noise = True
    return cfg


@configclass
class G1HostSupineEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _host(self, (270.0, 30.0))


@configclass
class G1HostProneEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _host(self, (90.0, 30.0))


@configclass
class G1HostSupineEvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _host(self, (270.0, 30.0), eval_mode=True)


@configclass
class G1HostProneEvalEnvCfg(G1RecoveryBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _host(self, (90.0, 30.0), eval_mode=True)


# ---------------------------------------------------------------- policy network (HoST ActorCritic: actor 512-256-128 ELU, tanh output; 4 critics 512-256)
def mlp(i, hidden, o, tanh=False):
    import torch.nn as nn
    L = []; d = i
    for hdim in hidden:
        L += [nn.Linear(d, hdim), nn.ELU()]; d = hdim
    L.append(nn.Linear(d, o))
    if tanh:
        L.append(nn.Tanh())
    return nn.Sequential(*L)


def load_host_policy(path, device):
    """Deterministic HoST actor from a train_host.py checkpoint: obs TensorDict/dict -> action mean."""
    ck = torch.load(path, map_location=device, weights_only=False)
    actor = mlp(ck["obs_dim"], ck["actor_hidden"], ck["act_dim"], tanh=True).to(device)
    actor.load_state_dict(ck["actor"]); actor.eval()
    return lambda obs: actor(obs["policy"])


def is_host_checkpoint(path):
    try:
        return "actor" in torch.load(path, map_location="cpu", weights_only=False) and "critics" in torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
