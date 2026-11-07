"""Lane A: pure task-reward recovery with a per-environment fallen-pose curriculum (promote on success, demote on no progress)."""
from __future__ import annotations
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
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
