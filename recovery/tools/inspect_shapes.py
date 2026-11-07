import sys, os
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
args = P._parse_args(sys.argv[1:])
from isaaclab_tasks.utils import resolve_task_config
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    m = NM.get_model()
    print("SHAPE_COUNT", m.shape_count, "BODY_COUNT", m.body_count, "JOINT_COUNT", m.joint_count)
    print([a for a in dir(m) if a.startswith("shape_")][:40])
    keys = getattr(m, "shape_label", None) or getattr(m, "shape_key", None)
    sb = m.shape_body.numpy(); st = m.shape_type.numpy() if hasattr(m,"shape_type") else None
    bl = getattr(m, "body_label", None) or getattr(m, "body_key", None)
    n1 = m.shape_count // env.unwrapped.num_envs
    for i in range(min(n1+2, m.shape_count)):
        print("SHAPE", i, "body", int(sb[i]), (bl[int(sb[i])] if bl is not None and sb[i]>=0 else None), "type", None if st is None else int(st[i]), (keys[i] if keys is not None else None))
    r = env.unwrapped.scene["robot"]
    print("BODY_NAMES", r.body_names)
    print("JOINT_NAMES", r.joint_names)
    print("DEFAULT_ROOT_H", float(r.data.default_root_state.torch[0,2]))
    print("SOFT_LIMITS", r.data.soft_joint_pos_limits.torch[0,:6].tolist())
    C.close_env(env)
