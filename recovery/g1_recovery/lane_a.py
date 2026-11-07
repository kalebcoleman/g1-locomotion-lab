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
