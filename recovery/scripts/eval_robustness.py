"""M4 (EVALUATION ONLY): push and floor-friction robustness of the frozen walking policy in its native env (Isaac-Velocity-Flat-G1, minimal-collision G1, same obs/action).
  python eval_robustness.py --task Isaac-Velocity-Flat-G1 --walk_checkpoint <walking.pt> --seed 1 --friction 1.0 --out r.json physics=newton_mjwarp
Protocol: robots walk at a constant forward command for --pre_s seconds, then ONE horizontal velocity impulse (magnitude m m/s, direction relative to the robot's heading at that moment,
8 directions 45 deg apart: 0 = forward, 90 = left, 180 = backward, 270 = right) is added to the base velocity (same mechanism as M2). A robot SURVIVES if it does not fall during the
following --after_s seconds; a fall is the environment's torso-contact termination or pelvis < 0.45 m / torso up < 0.5 for 10 consecutive steps. A robot that falls before the push is
counted as a fall (reported separately). Friction: static = dynamic friction of the floor AND of the robot bodies is set to --friction (training: floor 1.0, robot 0.8 / 0.6), so it is the effective foot-floor coefficient whatever the solver's combine rule. Magnitude 0 is the control."""
import sys, os, json, argparse, math
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out", required=True); ex.add_argument("--walk_checkpoint", required=True)
ex.add_argument("--seed", type=int, default=1); ex.add_argument("--reps", type=int, default=4); ex.add_argument("--friction", type=float, default=1.0)
ex.add_argument("--mags", default="0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4"); ex.add_argument("--n_dirs", type=int, default=8)
ex.add_argument("--fwd", type=float, default=0.8); ex.add_argument("--pre_s", type=float, default=3.0); ex.add_argument("--after_s", type=float, default=5.0)
ex.add_argument("--record_dir", default=None); ex.add_argument("--label", default="m4")
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
if xa.record_dir: args.enable_cameras = True
import g1_recovery  # noqa: F401
from g1_recovery import metrics as M
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

mags = [float(v) for v in xa.mags.split(",")]; dirs = [i * 360.0 / xa.n_dirs for i in range(xa.n_dirs)]
cells = [(m, d) for m in mags for d in dirs]; N = len(cells) * xa.reps
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.events.push_robot = None
cm = env_cfg.commands.base_velocity; cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
cm.ranges.lin_vel_x = (xa.fwd, xa.fwd); cm.ranges.lin_vel_y = (0.0, 0.0); cm.ranges.ang_vel_z = (0.0, 0.0)
tm = env_cfg.scene.terrain.physics_material; tm.static_friction = xa.friction; tm.dynamic_friction = xa.friction
# MuJoCo-style solvers combine two contacting geoms with the MAXIMUM of their friction values, so lowering only the floor does nothing while the robot's own material stays at 0.8:
# set the robot bodies' startup material (the event) and the sim default to the same coefficient, so 'friction' is the effective foot-floor value.
env_cfg.events.physics_material.params["static_friction_range"] = (xa.friction, xa.friction); env_cfg.events.physics_material.params["dynamic_friction_range"] = (xa.friction, xa.friction)
if getattr(env_cfg.sim, "physics_material", None) is not None: env_cfg.sim.physics_material.static_friction = xa.friction; env_cfg.sim.physics_material.dynamic_friction = xa.friction
env_cfg.scene.num_envs = N; args.num_envs = N
DT = 0.02; PRE = int(round(xa.pre_s / DT)); AFT = int(round(xa.after_s / DT)); STEPS = PRE + AFT
if xa.record_dir:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, eye=(0, -11, 9), lookat=(0, 0, 0.3))]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.record_dir, video_length=STEPS, video_interval=STEPS, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = xa.seed; env_cfg.episode_length_s = xa.pre_s + xa.after_s + 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; assert u.num_envs == N; r = u.scene["robot"]; org = u.scene.env_origins
    wcfg = handle_deprecated_rsl_rl_cfg(G1FlatPPORunnerCfg(), check_rsl_rl_version()); wcfg.device = agent_cfg.device
    wr = create_rsl_rl_runner(env, wcfg); wr.load(xa.walk_checkpoint, map_location=agent_cfg.device); pol = wr.get_inference_policy(device=dev)
    obs, _ = env.reset(); term = u.command_manager.get_term("base_velocity")
    mag_t = torch.tensor([c[0] for c in cells for _ in range(xa.reps)], device=dev); dir_t = torch.tensor([math.radians(c[1]) for c in cells for _ in range(xa.reps)], device=dev, dtype=torch.float32)
    fell = torch.zeros(N, dtype=torch.bool, device=dev); fell_step = torch.full((N,), -1, device=dev, dtype=torch.long); cnt = torch.zeros(N, dtype=torch.long, device=dev)
    for k in range(STEPS):
        if k == PRE:
            ok = mag_t > 0
            if ok.any():
                ids = torch.nonzero(ok).squeeze(-1); q = r.data.root_quat_w.torch[ids]; psi = torch.atan2(2 * (q[:, 0] * q[:, 1] + q[:, 3] * q[:, 2]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
                vel = r.data.root_vel_w.torch[ids].clone(); vel[:, 0] += mag_t[ids] * torch.cos(psi + dir_t[ids]); vel[:, 1] += mag_t[ids] * torch.sin(psi + dir_t[ids])
                r.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        term.vel_command_b[:] = 0.0; term.vel_command_b[:, 0] = xa.fwd
        with torch.inference_mode():
            obs, _, dones, _ = env.step(pol(obs))
        h = r.data.root_pos_w.torch[:, 2] - org[:, 2]; up = -r.data.projected_gravity_b.torch[:, 2]
        cnt = torch.where((h < M.P.fall_h) | (up < M.P.fall_up), cnt + 1, torch.zeros_like(cnt))
        new = (dones.bool() | (cnt >= 10)) & ~fell; fell_step = torch.where(new, torch.full_like(fell_step, k), fell_step); fell |= new
    fell_c = fell.cpu(); fs = fell_step.cpu(); mag_c = mag_t.cpu(); dir_c = dir_t.cpu()
    pre_fall = fell_c & (fs < PRE)
    res = dict(protocol="M4 push/friction robustness (evaluation only)", task=args.task, walk_checkpoint=xa.walk_checkpoint, seed=xa.seed, friction=xa.friction, num_envs=N, reps_per_cell=xa.reps,
               mags=mags, dirs_deg=dirs, pre_s=xa.pre_s, after_s=xa.after_s, fwd_cmd=xa.fwd, fell_before_push=int(pre_fall.sum()),
               per_env=dict(mag=[round(float(m_), 3) for m_ in mag_c], dir_deg=[round(math.degrees(float(d_))) for d_ in dir_c], fell=fell_c.int().tolist(), fall_step=fs.tolist()))
    json.dump(res, open(xa.out, "w"), indent=1); print("M4_JSON", json.dumps(dict(friction=xa.friction, n=N, fell=int(fell_c.sum()), fell_before_push=int(pre_fall.sum()))), flush=True)
    C.close_env(env)
