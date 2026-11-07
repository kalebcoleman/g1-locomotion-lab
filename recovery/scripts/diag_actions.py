"""What the policy actually commands, on real simulated states from a pose set (assist off, deterministic mean action):
raw action distribution, joint-target distribution, targets beyond joint limits (saturation), joint positions beyond soft/hard limits (rad), torque saturation.
With --compare_ckpt/--compare_scale it also evaluates a second checkpoint on the SAME states (last-action input rescaled) and reports the joint-target difference.
  python diag_actions.py --task G1-Recovery-R5-Eval-v0 --checkpoint c.pt --num_envs 256 --pose_set s.npz --out d.json [--compare_ckpt o.pt --compare_scale 0.5] physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out", required=True); ex.add_argument("--pose_set", required=True); ex.add_argument("--seconds", type=float, default=10.0)
ex.add_argument("--compare_ckpt", default=None); ex.add_argument("--compare_scale", type=float, default=None)
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False); env_cfg.fixed_poses_path = os.path.abspath(xa.pose_set)


def mlp(S, x):
    h = x
    for i in (0, 2, 4): h = torch.nn.functional.elu(h @ S[f"mlp.{i}.weight"].T + S[f"mlp.{i}.bias"])
    return h @ S["mlp.6.weight"].T + S["mlp.6.bias"]


with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]; d = r.data; dt = u.step_dt
    runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device); policy = runner.get_inference_policy(device=dev)
    f = np.load(xa.pose_set); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    term = u.action_manager.get_term("joint_pos"); scale = float(term._scale) if isinstance(term._scale, (int, float)) else float(torch.as_tensor(term._scale).mean())
    cmp_S = None
    if xa.compare_ckpt:
        cmp_S = {k: v.to(dev) for k, v in torch.load(xa.compare_ckpt, map_location="cpu", weights_only=True)["actor_state_dict"].items()}
    names = list(r.joint_names); body = torch.tensor([not K.is_hand_joint(n) for n in names], device=dev); nj = len(names)
    soft = d.soft_joint_pos_limits.torch[0]; hard = d.joint_pos_limits.torch[0] if hasattr(d, "joint_pos_limits") else None; tau_lim = d.joint_effort_limits.torch[0] if hasattr(d, "joint_effort_limits") else None
    obs, _ = env.reset(); steps = int(xa.seconds / dt)
    raw_all, cmp_diff = [], []
    tgt_out_soft = torch.zeros(nj, device=dev); tgt_out_hard = torch.zeros(nj, device=dev); tgt_excess_max = torch.zeros(nj, device=dev)
    q_soft = torch.zeros(nj, device=dev); q_hard = torch.zeros(nj, device=dev); q_ex_soft = torch.zeros(nj, device=dev); q_ex_hard = torch.zeros(nj, device=dev); tau_sat = torch.zeros(nj, device=dev)
    any_q_soft = torch.zeros(N, device=dev); any_q_hard = torch.zeros(N, device=dev); q_ex_sum = torch.zeros(nj, device=dev)
    for s in range(steps):
        with torch.inference_mode():
            a = policy(obs)
            if cmp_S is not None:
                x = obs["policy"].clone(); x[:, -37:] *= scale / xa.compare_scale      # last action expressed in the other policy's action units
                cmp_diff.append((mlp(cmp_S, x) * xa.compare_scale - a * scale)[:, body].abs().max(-1).values.cpu())
            obs, _, _, _ = env.step(a)
        raw_all.append(a[:, body].cpu())
        t = term.processed_actions; q = d.joint_pos.torch
        lo, hi = soft[:, 0], soft[:, 1]
        tgt_out_soft += ((t < lo) | (t > hi)).float().mean(0); ex = torch.maximum(lo - t, t - hi).clamp(min=0); tgt_excess_max = torch.maximum(tgt_excess_max, ex.max(0).values)
        qex = torch.maximum(lo - q, q - hi).clamp(min=0); q_soft += (qex > 0).float().mean(0); q_ex_soft = torch.maximum(q_ex_soft, qex.max(0).values); q_ex_sum += qex.sum(0)
        any_q_soft += ((qex > 0) & body).any(-1).float()
        if hard is not None:
            tgt_out_hard += ((t < hard[:, 0]) | (t > hard[:, 1])).float().mean(0)
            hex_ = torch.maximum(hard[:, 0] - q, q - hard[:, 1]).clamp(min=0); q_hard += (hex_ > 0).float().mean(0); q_ex_hard = torch.maximum(q_ex_hard, hex_.max(0).values); any_q_hard += ((hex_ > 0) & body).any(-1).float()
        if tau_lim is not None: tau_sat += (d.applied_torque.torch.abs() >= 0.95 * tau_lim[None]).float().mean(0)
    A = torch.cat(raw_all).abs(); bi = [i for i in range(nj) if body[i]]
    top = lambda v, k=6, nd=3: {names[i]: round(float(v[i]), nd) for i in sorted(bi, key=lambda j: -float(v[j]))[:k]}
    res = dict(task=args.task, checkpoint=args.checkpoint, action_scale=scale, pose_set=os.path.basename(xa.pose_set), num_envs=N, seconds=xa.seconds, hard_limits_available=hard is not None,
               raw_action_abs=dict(mean=float(A.mean()), p50=float(A.quantile(0.5)), p90=float(A.quantile(0.9)), p99=float(torch.as_tensor(A.flatten()[::7]).quantile(0.99)), max=float(A.max())),
               joint_target_offset_rad_abs=dict(mean=float(A.mean() * scale), p99=float(torch.as_tensor(A.flatten()[::7]).quantile(0.99) * scale)),
               targets_beyond_soft_limit_frac_mean=float((tgt_out_soft[body] / steps).mean()), targets_beyond_soft_top=top(tgt_out_soft / steps), target_excess_beyond_soft_max_rad_top=top(tgt_excess_max),
               targets_beyond_hard_limit_frac_mean=float((tgt_out_hard[body] / steps).mean()) if hard is not None else None,
               q_beyond_soft_step_frac_any_joint_mean=float((any_q_soft / steps).mean()), q_beyond_soft_top=top(q_soft / steps), q_excess_soft_max_rad_top=top(q_ex_soft), q_excess_soft_mean_rad_when_counted=float(q_ex_sum[body].sum() / max(1.0, float((q_soft[body] * N).sum()))),
               q_beyond_hard_step_frac_any_joint_mean=float((any_q_hard / steps).mean()) if hard is not None else None, q_beyond_hard_top=top(q_hard / steps) if hard is not None else None, q_excess_hard_max_rad_top=top(q_ex_hard) if hard is not None else None,
               torque_sat_95pct_top=top(tau_sat / steps) if tau_lim is not None else None)
    if cmp_S is not None:
        D = torch.cat(cmp_diff); res["compare"] = dict(ckpt=xa.compare_ckpt, scale=xa.compare_scale, target_diff_rad_max=float(D.max()), target_diff_rad_mean=float(D.mean()), note="max over body joints of |other_policy*other_scale - this_policy*this_scale| on the same simulated states")
    json.dump(res, open(xa.out, "w"), indent=1); print("ACTDIAG_OK", flush=True); C.close_env(env)
