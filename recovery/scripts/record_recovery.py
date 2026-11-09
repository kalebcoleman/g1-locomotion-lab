"""Record a 10 s video of N robots starting from the first N held-out fallen poses (same poses for every policy).
  python record_recovery.py --task G1-Recovery-A-v0 --checkpoint <model.pt> --out_dir <abs dir> --label a_final --num_envs 16 physics=newton_mjwarp
Falls/resets: episodes are 20 s so no env resets during the 10 s clip; terminal stats printed as REC_JSON."""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out_dir", required=True); ex.add_argument("--label", default="rec")
ex.add_argument("--level", type=int, default=None, help="pin a training curriculum level (random start poses) instead of the held-out poses")
ex.add_argument("--eye", default="0,-11,9"); ex.add_argument("--lookat", default="0,0,0.3")
ex.add_argument("--heldout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heldout_poses.npz"))
xa, rest = ex.parse_known_args(sys.argv[1:])
eye = tuple(float(v) for v in xa.eye.split(",")); lookat = tuple(float(v) for v in xa.lookat.split(","))
args = P._parse_args(rest); args.train_env_cfg = True; args.enable_cameras = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = None if xa.level is not None else os.path.abspath(xa.heldout)
CYC = 500
env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, eye=eye, lookat=lookat)]
env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.out_dir, video_length=CYC, video_interval=CYC, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1; env_cfg.episode_length_s = 20.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device
    from g1_recovery import host_port as _HP
    if _HP.is_host_checkpoint(args.checkpoint):
        policy = _HP.load_host_policy(args.checkpoint, dev)      # HoST-port actor (train_host.py)
    else:
        runner = create_rsl_rl_runner(env, agent_cfg)
        if args.checkpoint: runner.load(args.checkpoint, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=dev)
    if xa.level is None:
        f = np.load(xa.heldout); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    else:
        K.get_levels(u)[:] = xa.level; u._rec_level_fixed = True
    obs, _ = env.reset(); t = K.get_tracker(u)
    for s in range(CYC):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
    fc = t.failure_cause()
    print("REC_JSON", json.dumps(dict(label=xa.label, num_envs=u.num_envs, successes=int(t.succ.sum()), longest_stand_s_mean=float(t.best_run.float().mean() * u.step_dt),
          failure_causes={n: int((fc == i).sum()) for i, n in enumerate(K.Tracker.FAIL)})), flush=True)
    C.close_env(env)
