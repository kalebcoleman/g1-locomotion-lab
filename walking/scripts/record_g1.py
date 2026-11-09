"""Record scripted-command videos of a G1 policy from several camera angles (Isaac Lab 3.0, headless Kit viewport).

One env, one 8 s cycle per camera angle: stand 2 s | walk fwd 2 s | turn 2 s | stop 2 s.
The camera follows the robot. Raw MP4s go to <out_dir>/<label>_<index>.mp4, one per angle, in the order of ANGLES.
Usage (run from the Isaac Lab checkout):
  python record_g1.py --task Isaac-Velocity-Flat-G1 --checkpoint X.pt --label newton_best --out_dir <dir> [--push_vel 0.5] physics=newton_mjwarp
"""
import sys, argparse, json
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

# (name, eye offset from robot root, lookat offset). Robot walks along +x (world, starting heading).
ANGLES = [
    ("front",         (3.0, 0.0, 0.6),   (0.0, 0.0, 0.0)),
    ("side",          (0.0, -3.0, 0.4),  (0.0, 0.0, 0.0)),
    ("threequarter",  (2.4, -2.4, 1.0),  (0.0, 0.0, -0.1)),
    ("top",           (-1.2, 0.0, 4.0),  (0.0, 0.0, 0.0)),
    ("feet_closeup",  (1.1, -0.9, -0.45),(0.0, 0.0, -0.65)),
]
ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--label", required=True)
ex.add_argument("--out_dir", required=True)
ex.add_argument("--push_vel", type=float, default=0.0)
ex.add_argument("--fwd", type=float, default=0.8)
ex.add_argument("--yaw", type=float, default=0.8)
ex.add_argument("--angles", default=",".join(a[0] for a in ANGLES))
xa, rest = ex.parse_known_args(sys.argv[1:])
angles = [a for a in ANGLES if a[0] in xa.angles.split(",")]

args = P._parse_args(rest)
args.enable_cameras = True  # loads omni.replicator, needed by the viewport frame grabber
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=not args.train_env_cfg)
CYC = 400
env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=960, window_height=540, eye=(3, 0, 0.6), lookat=(0, 0, 0), origin_type="asset", origin_track_path="robot")]
env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.out_dir, video_length=CYC,
                                            video_interval=CYC, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1
    cm = env_cfg.commands.base_velocity
    cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
    env_cfg.episode_length_s = 1000.0   # no timeouts; falls still terminate/reset the env
    if xa.push_vel <= 0:
        env_cfg.events.push_robot = None
    else:
        env_cfg.events.push_robot.interval_range_s = (2.0, 2.0)
        env_cfg.events.push_robot.params["velocity_range"] = {"x": (-xa.push_vel, xa.push_vel), "y": (-xa.push_vel, xa.push_vel)}
    print("PHYSICS:", type(env_cfg.sim.physics).__name__, getattr(env_cfg.sim.physics, "class_type", None))
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped; dev = u.device
    term = u.command_manager.get_term("base_velocity")
    obs = env.get_observations()
    seg = [(0, 0, 0), (xa.fwd, 0, 0), (0, 0, xa.yaw), (0, 0, 0)]
    log = {"label": xa.label, "angles": [], "falls": 0}
    for name, eo, lo in angles:
        falls = 0
        for s in range(CYC):
            c = torch.tensor(seg[s * 4 // CYC], device=dev, dtype=torch.float)
            term.vel_command_b[:] = c
            if s == 0:  # the Kit visualizer tracks the robot root; eye/lookat are offsets from it
                viz = u.sim.visualizers[0]; viz.cfg.eye = eo; viz.cfg.lookat = lo
            with torch.inference_mode():
                obs, _, dones, ext = env.step(policy(obs))
            falls += int(dones[0])
        log["angles"].append({"angle": name, "falls_in_clip": falls})
        print("CLIP_DONE", name, "falls", falls, flush=True)
    print("RECORD_JSON " + json.dumps(log))
    C.close_env(env)
