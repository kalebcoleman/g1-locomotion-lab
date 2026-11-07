"""Harvest a start-state bank by rolling out a recovery policy from TRAINING-distribution lying poses (level 4, seed != held-out).
Saves root height, quaternion and joint positions at 2/4/6/8/10 s for envs whose pelvis is above the floor (h in 0.12..0.62 m). Bank used for levels 1/2.
  python harvest_bank.py --task G1-Recovery-A-v0 --checkpoint <model.pt> --num_envs 2048 --out bank_v1.npz physics=newton_mjwarp"""
import sys, os, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out", required=True); ex.add_argument("--harvest_seed", type=int, default=777)
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = xa.harvest_seed; env_cfg.episode_length_s = 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; r = u.scene["robot"]
    runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device); policy = runner.get_inference_policy(device=dev)
    K.get_levels(u)[:] = 4; u._rec_bank = None
    u._rec_level_fixed = True
    obs, _ = env.reset()
    Z, Q, J = [], [], []
    for s in range(1, 501):
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        if s % 100 == 0:
            z = r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2]; qd = r.data.root_quat_w.torch
            ok = (z > 0.12) & (z < 0.62) & torch.isfinite(z)
            Z.append(z[ok].cpu()); Q.append(qd[ok].cpu()); J.append(r.data.joint_pos.torch[ok].cpu())
    z = torch.cat(Z).numpy(); q = torch.cat(Q).numpy(); j = torch.cat(J).numpy()
    np.savez(xa.out, z=z, quat=q, q=j, source=args.checkpoint, harvest_seed=xa.harvest_seed)
    up = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    print("BANK_SAVED", xa.out, "n", len(z), "sit(up>=0.7)", int((up >= 0.7).sum()), "z mean", float(z.mean()), flush=True)
    C.close_env(env)
