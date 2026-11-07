"""Eval-mode check: with cfg.fixed_poses_path set (as eval/record scripts do) the assist scale must stay 0 and no force is applied. Also train-mode default = 1.0."""
import sys, os
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
args = P._parse_args(sys.argv[1:]); args.train_env_cfg = True
import g1_recovery
from isaaclab_tasks.utils import resolve_task_config
env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = os.path.abspath("heldout_poses.npz")
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False); u = env.unwrapped
    env.reset()
    for _ in range(30): env.step(torch.zeros(u.num_envs, u.action_manager.total_action_dim, device=u.device))
    print("EVALCHECK scale", u._assist_scale, "max_force", float(getattr(u, "_assist_f", torch.zeros(1)).abs().max()), "assist_was_on", getattr(u, "_assist_was_on", False), flush=True)
    C.close_env(env)
