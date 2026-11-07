"""Create the versioned evaluation pose sets (needs the robot's default pose and joint limits, so it starts the sim briefly with 2 envs).
  python make_sets.py --out_dir <dir> physics=newton_mjwarp   -> heldout_v2.npz (seed 20261107), test_v2.npz (seed 20261108); DEV = the original heldout_poses.npz (seed 20261106)"""
import sys, os, argparse
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out_dir", required=True)
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    r = env.unwrapped.scene["robot"]; dev = env.unwrapped.device
    dq = r.data.default_joint_pos.torch[0]; lim = r.data.soft_joint_pos_limits.torch[0]
    mask = torch.tensor([0.0 if K.is_hand_joint(n) else 1.0 for n in r.joint_names], device=dev)
    for name, seed in (("heldout_v2", 20261107), ("test_v2", 20261108)):
        d = K.make_set(seed, dq, lim[:, 0], lim[:, 1], mask, path=os.path.join(xa.out_dir, name + ".npz"))
        print("MADE", name, seed, {k: (v.shape if hasattr(v, "shape") else v) for k, v in d.items()}, flush=True)
    C.close_env(env)
