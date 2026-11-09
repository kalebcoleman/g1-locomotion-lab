"""Measure link heights: standing default pose and in a given pose rollout (quick reference for reward targets). probe_heights.py --task T --num_envs 8 [--checkpoint c.pt --pose_set s.npz] physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--pose_set", default=None); xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False); env_cfg.fixed_poses_path = os.path.abspath(xa.pose_set) if xa.pose_set else None
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False); env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; r = u.scene["robot"]; dev = u.device
    runner = create_rsl_rl_runner(env, agent_cfg)
    if args.checkpoint: runner.load(args.checkpoint, map_location=agent_cfg.device)
    pol = runner.get_inference_policy(device=dev)
    if xa.pose_set:
        f = np.load(xa.pose_set); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    else:
        K.get_levels(u)[:] = 0; u._rec_level_fixed = True
    obs, _ = env.reset()
    names = list(r.body_names); idx = {n: names.index(n) for n in names if n in ("pelvis", "torso_link", "head_link", "imu_link") or "ankle_roll" in n}
    for s in range(250):
        with torch.inference_mode(): obs, _, _, _ = env.step(pol(obs))
        if s in (0, 50, 249):
            z = r.data.body_pos_w.torch[:, :, 2] - u.scene.env_origins[:, 2:3]; g = r.data.projected_gravity_b.torch[:, 2]
            q = r.data.body_quat_w.torch[:, idx["torso_link"]]; tup = 1 - 2 * (q[:, 0] ** 2 + q[:, 1] ** 2)
            print("HEIGHTS step", s, {n: round(float(z[:, i].mean()), 3) for n, i in idx.items()}, "pelvis_up", round(float(-g.mean()), 3), "torso_up", round(float(tup.mean()), 3), flush=True)
    C.close_env(env)
