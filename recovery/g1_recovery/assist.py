"""HoST-style (Huang et al., arXiv:2502.08378, MIT; re-implemented, no code copied) gated, per-environment assistive pulling force.

Differences from HoST are listed in lanes/a/NOTES_assist.md. Behaviour:
* a vertical +z force on torso_link of ASSIST_FRAC * level * robot weight,
* applied only while the root torso is already fairly upright (projected gravity z < -0.8, like HoST) AND the pelvis is below CAP_H (so a
  standing robot is never held up by the force; HoST has no height cap),
* `level` is PER ENV in [0, 1]; at each reset the env's level drops by DECAY if its finished episode reached the standing criterion at least once
  (HoST: -20 N of 100 N when the env's mean head height beats a threshold); it never increases,
* everything is OFF unless env var G1R_ASSIST_TRAIN=1 (training launches) or G1R_EVAL_ASSIST=<level> (evaluation with a fixed level); default 0.
The force actually applied and when is logged in env._assist_log for reporting.
"""
from __future__ import annotations
import os
import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from . import common as K

ASSIST_FRAC = 0.6
UPRIGHT_MIN = 0.8          # = projected_gravity_z < -0.8
CAP_H = 0.62               # m, no force above this pelvis height
DECAY = 0.2                # per successful episode (5 successes -> 0)
TORSO = "torso_link"


def _level(env):
    if not hasattr(env, "_assist_level"):
        init = 0.0
        if os.environ.get("G1R_ASSIST_TRAIN", "0") == "1" or getattr(env.cfg, "assist_train", False):
            init = 1.0
        if os.environ.get("G1R_EVAL_ASSIST") is not None:
            init = float(os.environ["G1R_EVAL_ASSIST"])
        env._assist_level = torch.full((env.num_envs,), init, device=env.device)
        env._assist_fixed = os.environ.get("G1R_EVAL_ASSIST") is not None
        n = env.num_envs
        env._assist_log = dict(force_steps=torch.zeros(n, device=env.device), impulse=torch.zeros(n, device=env.device),
                               first=torch.full((n,), -1.0, device=env.device), last=torch.full((n,), -1.0, device=env.device),
                               max_force=torch.zeros(n, device=env.device))
        env._assist_on_flag = False
    return env._assist_level


def assist_hook(env, env_ids=None):
    """Per-step hook in the termination slot (never terminates). Sets the force for the next step."""
    r = env.scene["robot"]; lvl = _level(env)
    on = bool((lvl > 0).any())
    if on or env._assist_on_flag:
        if not hasattr(env, "_torso_bid"):
            env._torso_bid = r.find_bodies(TORSO)[0][0]
            env._robot_weight = float(r.data.body_mass.torch[0].sum()) * 9.81
        h = r.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        up = -r.data.projected_gravity_b.torch[:, 2]
        gate = (up >= UPRIGHT_MIN) & (h < CAP_H)
        fz = ASSIST_FRAC * lvl * env._robot_weight * gate.float()
        f = torch.zeros(env.num_envs, 1, 3, device=env.device); f[:, 0, 2] = fz
        r.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=torch.zeros_like(f), body_ids=[env._torso_bid], env_ids=None, is_global=True)
        L = env._assist_log; step = float(env.common_step_counter)
        act = fz > 0
        L["force_steps"] += act.float(); L["impulse"] += fz * env.step_dt; L["max_force"] = torch.maximum(L["max_force"], fz)
        L["first"] = torch.where(act & (L["first"] < 0), torch.full_like(L["first"], step), L["first"]); L["last"] = torch.where(act, torch.full_like(L["last"], step), L["last"])
        env._assist_on_flag = on
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def assist_decay(env, env_ids):
    """Curriculum term (runs at reset BEFORE the tracker is cleared): per-env level decay on reaching the standing criterion."""
    lvl = _level(env)
    if env._assist_fixed or isinstance(env_ids, slice) or len(env_ids) == 0 or env.common_step_counter == 0:
        return lvl.mean()
    tr = K.get_tracker(env); mode = getattr(env.cfg, "assist_decay_on", "legacy")
    if mode == "supported1s":                       # R3: decay only after a SUPPORTED stand held >= 1 s (50 steps), never for a transient pelvis-height crossing
        ok = (tr.best_run_s >= 50)[env_ids]
    else:
        ok = (tr.ever_stood_s if K.STRICT else tr.ever_stood)[env_ids]
    lvl[env_ids] = torch.clamp(lvl[env_ids] - getattr(env.cfg, "assist_decay", DECAY) * ok.float(), 0.0, 1.0)
    return lvl.mean()


def add_assist(cfg):
    """Attach the hook + decay term to an env cfg (training and evaluation share the same cfg; env vars decide whether it is active)."""
    cfg.terminations.assist = DoneTerm(func=assist_hook)
    cfg.curriculum.assist_level = CurrTerm(func=assist_decay)
    return cfg
