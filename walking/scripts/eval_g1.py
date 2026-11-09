"""Scripted-command evaluation of an rsl_rl G1 velocity policy (Isaac Lab 3.0).

Phases (6 s each): stand | walk fwd | turn in place | stop.  Pushes can be enabled.
Usage (run from the Isaac Lab checkout; Lab 3 is headless by default, so no --headless flag):
  python <this> --task Isaac-Velocity-Flat-G1 --checkpoint X.pt --num_envs 256 \
     --train_env_cfg --push_vel 0.5 --push_interval 2.5 [physics=newton_mjwarp]
"""
import sys, json, time, argparse
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

# extra args are stripped before the launcher parser sees them
ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--push_vel", type=float, default=0.0)
ex.add_argument("--push_interval", type=float, default=2.5)
ex.add_argument("--fwd", type=float, default=0.8)
ex.add_argument("--yaw", type=float, default=0.8)
ex.add_argument("--phase_s", type=float, default=6.0)
ex.add_argument("--out", type=str, default=None)
xa, rest = ex.parse_known_args(sys.argv[1:])

args = P._parse_args(rest)
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=not args.train_env_cfg)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = args.seed if args.seed is not None else 1
    cm = env_cfg.commands.base_velocity
    cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
    env_cfg.episode_length_s = 4 * xa.phase_s + 5
    if xa.push_vel > 0:
        from isaaclab.managers import EventTermCfg
        if env_cfg.events.push_robot is not None:
            env_cfg.events.push_robot.interval_range_s = (xa.push_interval, xa.push_interval)
            env_cfg.events.push_robot.params["velocity_range"] = {"x": (-xa.push_vel, xa.push_vel), "y": (-xa.push_vel, xa.push_vel)}
        else:
            print("WARNING: no push_robot event"); 
    else:
        env_cfg.events.push_robot = None
    print("PHYSICS:", type(env_cfg.sim.physics).__name__, getattr(env_cfg.sim.physics, "class_type", None))
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg)
    ck = C.retrieve_file_path(args.checkpoint) if hasattr(C, "retrieve_file_path") else args.checkpoint
    runner.load(ck)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped; dev = u.device; N = u.num_envs
    term = u.command_manager.get_term("base_velocity")
    obs = env.get_observations()
    dt = u.step_dt; spp = int(round(xa.phase_s / dt))
    cmds = [(0, 0, 0), (xa.fwd, 0, 0), (0, 0, xa.yaw), (0, 0, 0)]
    names = ["stand", "walk", "turn", "stop"]
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    res = {}
    t0 = time.time(); steps = 0
    for pi, c in enumerate(cmds):
        errs = []; fell_in_phase = 0
        cc = torch.tensor(c, device=dev, dtype=torch.float)
        for s in range(spp):
            term.vel_command_b[:] = cc
            with torch.inference_mode():
                obs, _, dones, ext = env.step(policy(obs))
            steps += 1
            to = ext.get("time_outs", torch.zeros_like(dones)).bool() if isinstance(ext, dict) else torch.zeros_like(dones).bool()
            fall = dones.bool() & ~to & alive
            fell_in_phase += int(fall.sum()); alive &= ~fall
            # tracking error (only envs not yet fallen and not just reset)
            lv = u.scene["robot"].data.root_lin_vel_b.torch[:, :2]; av = u.scene["robot"].data.root_ang_vel_b.torch[:, 2]
            if s > spp // 3:  # skip transient
                e = torch.cat([lv - cc[:2], (av - cc[2]).unsqueeze(1)], 1)
                m = alive & ~dones.bool()
                if m.any(): errs.append(torch.stack([e[m, :2].norm(dim=1).mean(), e[m, 2].abs().mean()]))
        er = torch.stack(errs).mean(0).tolist() if errs else [float("nan")] * 2
        res[names[pi]] = dict(cmd=c, new_falls=fell_in_phase, survivors=int(alive.sum()), lin_err=er[0], yaw_err=er[1])
    wall = time.time() - t0
    res["N"] = N; res["steps_per_env"] = steps; res["wall_s"] = wall
    res["env_steps_per_s"] = N * steps / wall
    res["push"] = [xa.push_vel, xa.push_interval]
    res["survive_frac"] = float(alive.float().mean())
    print("RESULT_JSON " + json.dumps(res))
    if xa.out: json.dump(res, open(xa.out, "w"), indent=1)
    C.close_env(env)
