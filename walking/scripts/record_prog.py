"""Training-progression clips: the SAME scripted rollout for a ladder of checkpoints, one process, one MP4 per checkpoint.

N robots, fixed world camera, 10 s: stand 2.5 s | walk fwd 2.5 s | turn 2.5 s | stop 2.5 s. No pushes. Same seed before every reset.
Clips are written in the order of --ckpts to <out_dir>/<label>_<index>.mp4; a JSON with the index -> checkpoint mapping is printed.
Usage (run from the Isaac Lab checkout):
  python record_prog.py --task Isaac-Velocity-Flat-G1 --run_dir logs/rsl_rl/g1_flat/<run> --ckpts 0,50,100 --label prog --out_dir D physics=newton_mjwarp
"""
import sys, argparse, json, os
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--run_dir", required=True)
ex.add_argument("--ckpts", required=True)
ex.add_argument("--label", default="prog")
ex.add_argument("--out_dir", required=True)
ex.add_argument("--eye", default="0,-11,9")
ex.add_argument("--lookat", default="0,0,0.3")
ex.add_argument("--fwd", type=float, default=0.8)
ex.add_argument("--yaw", type=float, default=0.8)
xa, rest = ex.parse_known_args(sys.argv[1:])
ckpts = [int(c) for c in xa.ckpts.split(",")]
eye = tuple(float(v) for v in xa.eye.split(",")); lookat = tuple(float(v) for v in xa.lookat.split(","))

args = P._parse_args(rest)
args.enable_cameras = True
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=not args.train_env_cfg)
CYC = 500
env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, eye=eye, lookat=lookat)]
env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.out_dir, video_length=CYC,
                                            video_interval=CYC, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1
    cm = env_cfg.commands.base_velocity
    cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
    env_cfg.episode_length_s = 1000.0
    env_cfg.events.push_robot = None
    print("PHYSICS:", type(env_cfg.sim.physics).__name__, getattr(env_cfg.sim.physics, "class_type", None))
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg)
    u = env.unwrapped; dev = u.device
    term = u.command_manager.get_term("base_velocity")
    seg = [(0, 0, 0), (xa.fwd, 0, 0), (0, 0, xa.yaw), (0, 0, 0)]
    out = []
    for idx, it in enumerate(ckpts):
        runner.load(os.path.join(xa.run_dir, f"model_{it}.pt"))
        policy = runner.get_inference_policy(device=dev)
        torch.manual_seed(1)
        obs, _ = env.reset()
        falls = 0
        for s in range(CYC):
            term.vel_command_b[:] = torch.tensor(seg[s * 4 // CYC], device=dev, dtype=torch.float)
            with torch.inference_mode():
                obs, _, dones, _ = env.step(policy(obs))
            falls += int(dones.sum())
        out.append({"index": idx, "iteration": it, "terminations_in_clip": falls, "num_envs": u.num_envs})
        print("CLIP_DONE", it, "terminations", falls, flush=True)
    print("PROG_JSON " + json.dumps(out))
    C.close_env(env)
