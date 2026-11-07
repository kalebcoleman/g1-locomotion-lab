"""Walking-only baseline on either robot model, same protocol: 3 s settle with zero command, then a 10 s window with vx=0.8 m/s (optionally then an 8 s yaw-turn at 0.8 rad/s), no pushes.
  python eval_walk_only.py --task Isaac-Velocity-Flat-G1 (minimal collision) | G1-Recovery-A-v0 (full collision, level-0 standing start) --checkpoint walking.pt --num_envs 256 --out r.json [--push 0.0]"""
import sys, os, json, argparse, math
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out", required=True); ex.add_argument("--push", type=float, default=0.0)
ex.add_argument("--fwd", type=float, default=0.8); ex.add_argument("--yaw", type=float, default=0.8)
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K, metrics as M
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
is_rec = args.task.startswith("G1-Recovery")
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 60.0
    cm = env_cfg.commands.base_velocity; cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
    if is_rec: env_cfg.pose_mix = None      # walking test: level-0 standing start, whatever the task's training start mix
    if not is_rec:
        if xa.push > 0 and env_cfg.events.push_robot is not None:
            env_cfg.events.push_robot.interval_range_s = (2.5, 2.5); env_cfg.events.push_robot.params["velocity_range"] = {"x": (-xa.push, xa.push), "y": (-xa.push, xa.push)}
        else: env_cfg.events.push_robot = None
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]; dt = u.step_dt
    runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device); pol = runner.get_inference_policy(device=dev)
    if is_rec: K.get_levels(u)[:] = 0; u._rec_level_fixed = True
    obs, _ = env.reset(); term = u.command_manager.get_term("base_velocity"); org = u.scene.env_origins[:, :2]
    S, W, TURN = int(3 / dt), int(10 / dt), int(8 / dt)
    fell = torch.zeros(N, dtype=torch.bool, device=dev); trk = torch.zeros(N, device=dev)
    _names = list(r.joint_names); _body = torch.tensor([not K.is_hand_joint(n) for n in _names], device=dev); _soft = r.data.soft_joint_pos_limits.torch[0]; _hard = r.data.joint_pos_limits.torch[0]
    _st = dict(n=0, soft=torch.zeros(N, device=dev), hard=torch.zeros(N, device=dev), hmax=torch.zeros(N, device=dev), smax=torch.zeros(N, device=dev), hj=torch.zeros(len(_names), device=dev), sj=torch.zeros(len(_names), device=dev))
    def stepcmd(vx, wz):
        global obs
        term.vel_command_b[:] = 0.0; term.vel_command_b[:, 0] = vx; term.vel_command_b[:, 2] = wz
        with torch.inference_mode(): obs, _, dn, _ = env.step(pol(obs))
        q = r.data.joint_pos.torch; sx = torch.maximum(_soft[:, 0] - q, q - _soft[:, 1]).clamp(min=0) * _body; hx = torch.maximum(_hard[:, 0] - q, q - _hard[:, 1]).clamp(min=0) * _body
        _st["n"] += 1; _st["soft"] += (sx.max(-1).values > 0).float(); _st["hard"] += (hx.max(-1).values > 0).float(); _st["hmax"] = torch.maximum(_st["hmax"], hx.max(-1).values); _st["smax"] = torch.maximum(_st["smax"], sx.max(-1).values); _st["hj"] += (hx > 0).float().mean(0); _st["sj"] += (sx > 0).float().mean(0)
        h = r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]; up = -r.data.projected_gravity_b.torch[:, 2]
        return M.fell(h, up)
    for _ in range(S): fell |= stepcmd(0.0, 0.0)
    q = r.data.root_quat_w.torch; hx = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2); hy = 2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]); heading = torch.stack([hx, hy], -1)
    p0 = r.data.root_pos_w.torch[:, :2] - org; fell_w = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(W):
        fell_w |= stepcmd(xa.fwd, 0.0); trk += (r.data.root_lin_vel_b.torch[:, 0] - xa.fwd).abs()
    p1 = r.data.root_pos_w.torch[:, :2] - org; along, lat = M.heading_progress(p0, p1, heading)
    yaw0 = torch.atan2(2 * (r.data.root_quat_w.torch[:, 3] * r.data.root_quat_w.torch[:, 2]), 1 - 2 * r.data.root_quat_w.torch[:, 2] ** 2); fell_t = torch.zeros(N, dtype=torch.bool, device=dev); yaw_sum = torch.zeros(N, device=dev)
    for _ in range(TURN):
        fell_t |= stepcmd(0.0, xa.yaw); yaw_sum += r.data.root_ang_vel_w.torch[:, 2] * dt
    ok_walk = ~fell & ~fell_w & (along >= M.P.walk_dist) & (along / 10.0 >= M.P.walk_speed)
    res = dict(task=args.task, robot_model="full-collision g1.usd" if is_rec else "minimal-collision g1_minimal.usd", checkpoint=args.checkpoint, num_envs=N, pushes_m_s=xa.push,
               protocol="3 s settle (zero cmd), 10 s walk vx=%.2f, 8 s turn wz=%.2f; walk ok = no fall AND >=3 m along heading AND >=0.5 m/s" % (xa.fwd, xa.yaw),
               fell_in_settle=int(fell.sum()), fell_in_walk=int(fell_w.sum()), fell_in_turn=int(fell_t.sum()), walk_successes=int(ok_walk.sum()), walk_success_rate=float(ok_walk.float().mean()),
               along_disp_m_mean=float(along.mean()), along_disp_m_p10=float(along.quantile(0.1)), lateral_m_mean=float(lat.mean()), fwd_tracking_abs_err_mean=float((trk / W).mean()),
               turn_yaw_rad_mean=float(yaw_sum.mean()), turn_yaw_target_rad=xa.yaw * 8.0, turn_success_ge_90deg=int(((yaw_sum.abs() >= math.pi / 2) & ~fell_t).sum()))
    _top = lambda v: {_names[i]: round(float(v[i] / _st["n"]), 4) for i in torch.argsort(v, descending=True)[:5].tolist()}
    res["joint_limits"] = dict(soft_step_frac_any_joint_mean=float((_st["soft"] / _st["n"]).mean()), soft_excess_rad_mean_of_env_max=float(_st["smax"].mean()), soft_excess_rad_max=float(_st["smax"].max()), soft_top=_top(_st["sj"]),
                               hard_step_frac_any_joint_mean=float((_st["hard"] / _st["n"]).mean()), hard_excess_rad_mean_of_env_max=float(_st["hmax"].mean()), hard_excess_rad_p99=float(_st["hmax"].quantile(0.99)), hard_excess_rad_max=float(_st["hmax"].max()), hard_top=_top(_st["hj"]))
    json.dump(res, open(xa.out, "w"), indent=1); print("WALK_JSON", json.dumps(res), flush=True); C.close_env(env)
