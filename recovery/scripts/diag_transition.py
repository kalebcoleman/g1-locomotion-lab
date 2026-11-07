"""Diagnose the stuck sit->stand transition: roll a policy from level-N start states and log what the body is doing.
Per step: pelvis height, torso-up, per-body net contact force (feet / hands / other), joint torque saturation, joint-limit hits per joint,
reward-term means (Lane A terms). Output JSON summary. Read-only analysis; unassisted (no curriculum runs).
  python diag_transition.py --task G1-Recovery-A-v0 --checkpoint m.pt --level 1 --num_envs 256 --seconds 6 --out d.json physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out", required=True); ex.add_argument("--level", type=int, default=1); ex.add_argument("--seconds", type=float, default=6.0)
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from g1_recovery import lane_a
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = None
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 5; env_cfg.episode_length_s = 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]
    runner = create_rsl_rl_runner(env, agent_cfg)
    if args.checkpoint: runner.load(args.checkpoint, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=dev)
    K.get_levels(u)[:] = xa.level; u._rec_level_fixed = True; u._assist_scale = 0.0
    obs, _ = env.reset()
    sens = u.scene.sensors.get("contact_forces") if hasattr(u.scene, "sensors") else None
    bnames = list(sens.body_names) if sens is not None else []
    def groups():
        feet = [i for i, n in enumerate(bnames) if "ankle_roll" in n]
        hands = [i for i, n in enumerate(bnames) if any(k in n for k in ("wrist", "elbow", "hand", "palm", "zero_link", "one_link", "two_link", "three_link", "four_link", "five_link", "six_link"))]
        legs = [i for i, n in enumerate(bnames) if any(k in n for k in ("knee", "hip", "ankle_pitch"))]
        trunk = [i for i, n in enumerate(bnames) if any(k in n for k in ("pelvis", "torso", "head", "imu"))]
        return dict(feet=feet, hands=hands, legs_non_foot=legs, trunk=trunk)
    G = groups(); steps = int(xa.seconds / u.step_dt)
    acc = {k: [] for k in ["h", "up", "feet_f", "hands_f", "legs_f", "trunk_f", "tau_sat", "limit_frac", "feet_contact_frac", "r_upright", "r_height", "r_stand", "r_pose"]}
    tau_lim = r.data.joint_effort_limits.torch[0] if hasattr(r.data, "joint_effort_limits") else None
    lim = r.data.soft_joint_pos_limits.torch; nj = len(r.joint_names); hits = torch.zeros(nj, device=dev); body_mask = torch.tensor([not K.is_hand_joint(n) for n in r.joint_names], device=dev)
    vel_lim = r.data.soft_joint_vel_limits.torch[0] if hasattr(r.data, "soft_joint_vel_limits") else None
    peak_tau_ratio = torch.zeros(nj, device=dev); peak_vel_ratio = torch.zeros(nj, device=dev); peak_vz = torch.zeros(N, device=dev); peak_h = torch.zeros(N, device=dev)
    for s in range(steps):
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        if tau_lim is not None: peak_tau_ratio = torch.maximum(peak_tau_ratio, (r.data.applied_torque.torch.abs() / tau_lim[None]).max(0).values)
        if vel_lim is not None: peak_vel_ratio = torch.maximum(peak_vel_ratio, (r.data.joint_vel.torch.abs() / vel_lim[None]).max(0).values)
        peak_vz = torch.maximum(peak_vz, r.data.root_lin_vel_w.torch[:, 2]); peak_h = torch.maximum(peak_h, r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2])
        h = r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]; up = -r.data.projected_gravity_b.torch[:, 2]
        acc["h"].append(h.cpu()); acc["up"].append(up.cpu())
        if sens is not None:
            f = torch.linalg.norm(sens.data.net_forces_w.torch, dim=-1)       # (N, B)
            for k, ix in G.items():
                key = {"feet": "feet_f", "hands": "hands_f", "legs_non_foot": "legs_f", "trunk": "trunk_f"}[k]
                acc[key].append((f[:, ix].sum(-1) if ix else torch.zeros(N, device=dev)).cpu())
            acc["feet_contact_frac"].append((f[:, G["feet"]] > 10.0).float().mean(-1).cpu() if G["feet"] else torch.zeros(N))
        tq = r.data.applied_torque.torch
        if tau_lim is not None: acc["tau_sat"].append(((tq.abs() >= 0.95 * tau_lim[None]) & body_mask[None]).float().sum(-1).cpu())
        q = r.data.joint_pos.torch; out = ((q < lim[..., 0]) | (q > lim[..., 1])) & body_mask[None]
        hits += out.float().mean(0); acc["limit_frac"].append(out.float().sum(-1).cpu())
        acc["r_upright"].append(lane_a.upright(u).cpu()); acc["r_height"].append(lane_a.height_ramp(u).cpu()); acc["r_stand"].append(lane_a.standing_bonus(u).cpu()); acc["r_pose"].append(lane_a.default_pose_when_upright(u).cpu())
    S = lambda k: torch.stack(acc[k]) if acc[k] else None     # (T, N)
    h = S("h"); res = dict(task=args.task, checkpoint=args.checkpoint, level=xa.level, num_envs=N, seconds=xa.seconds, bodies=bnames, groups={k: [bnames[i] for i in v] for k, v in G.items()})
    res["pelvis_h_m"] = dict(start_mean=float(h[0].mean()), end_mean=float(h[-1].mean()), end_p90=float(h[-1].quantile(0.9)), max_over_time_mean=float(h.max(0).values.mean()), frac_envs_ever_above_0p35=float((h.max(0).values > 0.35).float().mean()), frac_envs_ever_above_0p6=float((h.max(0).values > 0.6).float().mean()))
    res["torso_up_end_mean"] = float(S("up")[-1].mean())
    for k in ["feet_f", "hands_f", "legs_f", "trunk_f", "tau_sat", "limit_frac", "feet_contact_frac", "r_upright", "r_height", "r_stand", "r_pose"]:
        x = S(k)
        if x is not None: res[k + "_mean_over_time"] = float(x.mean())
    top = torch.argsort(hits, descending=True)[:6]; res["joint_limit_top"] = {r.joint_names[int(i)]: round(float(hits[int(i)] / steps), 3) for i in top}
    res["actuator_limits"] = dict(peak_torque_over_limit_by_joint={r.joint_names[i]: round(float(peak_tau_ratio[i]), 2) for i in torch.argsort(peak_tau_ratio, descending=True)[:8].tolist()}, peak_joint_vel_over_soft_limit_by_joint={r.joint_names[i]: round(float(peak_vel_ratio[i]), 2) for i in torch.argsort(peak_vel_ratio, descending=True)[:8].tolist()} if vel_lim is not None else None,
        pelvis_peak_vz_mean=float(peak_vz.mean()), pelvis_peak_vz_max=float(peak_vz.max()), pelvis_peak_height_mean=float(peak_h.mean()), pelvis_peak_height_max=float(peak_h.max()))
    res["note"] = "net contact force sums are in N; tau_sat = mean number of body joints at >=95% effort limit; limit_frac = mean number of body joints beyond soft limits"
    json.dump(res, open(xa.out, "w"), indent=1); print("DIAG_JSON", json.dumps({k: v for k, v in res.items() if k not in ("bodies", "groups")}), flush=True)
    C.close_env(env)
