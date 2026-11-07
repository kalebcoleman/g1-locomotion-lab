"""Probe: zero-action rollouts on held-out poses with assist scale 0 / 1 / 2: pelvis height trace, applied force, reward terms.
 run.sh path: lanes/b/probe_assist.py needs absolute path; use: scripts/run.sh ../lanes/b/probe_assist.py --task G1-Recovery-B-v0 --num_envs 64 physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
args = P._parse_args(sys.argv[1:]); args.train_env_cfg = True
import g1_recovery
from isaaclab_tasks.utils import resolve_task_config
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "heldout_poses.npz"))
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 20.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    u = env.unwrapped; N = u.num_envs; r = u.scene["robot"]
    print("PROBE default _assist_scale attr:", getattr(u, "_assist_scale", "unset (=0)"), "weight N:", float(r.data.body_mass.torch[0].sum()) * 9.81)
    for scale in [0.0, 1.0, 1.5]:
        u._assist_scale = scale; u.cfg.fixed_poses_path = None
        from g1_recovery import common as K
        K.get_levels(u).fill_(4)
        env.reset()
        a = torch.zeros(N, u.action_manager.total_action_dim, device=u.device)
        tr = []; fz = []
        for s in range(200):
            env.step(a)
            h = r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]
            tr.append(float(h.mean())); fz.append(float(getattr(u, "_assist_f", torch.zeros(1)).mean()))
        rew = {k: float(v) for k, v in zip(u.reward_manager.active_terms, u.reward_manager._step_reward.mean(0))}
        print("PROBE scale", scale, "pelvis_h@0,25,50,100,199:", [round(tr[i], 3) for i in (0, 25, 50, 100, 199)],
              "max_h", round(max(tr), 3), "mean_force_N@25,100:", round(fz[25], 1), round(fz[100], 1), flush=True)
        print("PROBE step rewards", json.dumps({k: round(v, 4) for k, v in rew.items()}), "finite:", all(np.isfinite(list(rew.values()))), flush=True)
    u._assist_scale = 0.0   # eval default check
    C.close_env(env)
