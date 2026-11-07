import sys, torch, warp as wp
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
args = P._parse_args(sys.argv[1:]); args.train_env_cfg = True
import g1_recovery
from isaaclab_tasks.utils import resolve_task_config
env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    r = env.unwrapped.scene["robot"]; d = r.data
    q = d.default_joint_pos.torch[0]; lim = d.soft_joint_pos_limits.torch[0]; hard = d.joint_pos_limits.torch[0] if hasattr(d, "joint_pos_limits") else None
    for i, n in enumerate(r.joint_names):
        flag = "VIOL" if (q[i] < lim[i, 0] or q[i] > lim[i, 1]) else ""
        print("JL", n, round(float(q[i]), 3), [round(float(x), 3) for x in lim[i]], flag)
    C.close_env(env)
