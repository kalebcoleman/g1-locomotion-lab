"""M3 (EVALUATION ONLY): the frozen flat-trained walking policy on graded terrain.
  python eval_terrain.py --task Isaac-Velocity-Rough-G1 --walk_checkpoint <walking.pt> --seed 1 --reps 6 --out r.json physics=newton_mjwarp
Env: Isaac Lab's Isaac-Velocity-Rough-G1 (the same minimal-collision G1 and 37-d action as the walking task) with the height scan REMOVED, so the policy sees the same 123-d observation it was
trained on (blind walking). Pushes are off; the velocity command is a constant forward 0.8 m/s; heading command is off. Every robot starts on the centre platform of its own sub-terrain
(random yaw and +-0.5 m offset from the standard reset event) and walks outward for --horizon_s seconds.
Terrain types (levels are rows; difficulty d = (row + U(0,1)) / rows, so a level covers [row/rows, (row+1)/rows)):
  slope_down / slope_up : pyramid / inverted pyramid slope, ratio = 0.0 + d * 0.5 (up to 26.6 deg); walking outward goes down / up
  steps_down / steps_up : pyramid / inverted pyramid stairs, step height = 0.02 + d * 0.12 m, step width 0.4 m
  rough_k               : random uniform ground, one column per noise amplitude (0.01 ... 0.14 m, noise step 0.01 m, 0.2 m grid); amplitude does not depend on the row
  flat                  : plane (reference)
A robot SURVIVES if the environment never terminates it (torso contact) during the horizon. Distance = displacement in the floor plane along the initial heading (distance reached before the
fall for robots that fall). Speed = distance / horizon for survivors."""
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
ex.add_argument("--seed", type=int, default=1); ex.add_argument("--reps", type=int, default=6); ex.add_argument("--levels", type=int, default=8)
ex.add_argument("--fwd", type=float, default=0.8); ex.add_argument("--horizon_s", type=float, default=10.0)
ex.add_argument("--record_dir", default=None); ex.add_argument("--label", default="m3")
ex.add_argument("--only", default=None, help="comma list of type:level cells to run (e.g. slope_up:2,steps_down:3) instead of all cells; used for the 16-robot video")
ex.add_argument("--cam", default="0,-60,50", help="camera eye offset from the centroid of the selected cells (video runs)")
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
if xa.record_dir: args.enable_cameras = True
import g1_recovery  # noqa: F401
from g1_recovery import metrics as M, robustness_stats as RS
from isaaclab import terrains as T
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

SLOPE_MAX = 0.5; STEP_RANGE = (0.02, 0.14); ROUGH_AMPS = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.14]
ROWS = xa.levels
subs = {"flat": T.MeshPlaneTerrainCfg(proportion=1.0),
        "slope_down": T.HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.0, SLOPE_MAX), platform_width=2.0, border_width=0.25),
        "slope_up": T.HfInvertedPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.0, SLOPE_MAX), platform_width=2.0, border_width=0.25),
        "steps_down": T.MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=STEP_RANGE, step_width=0.4, platform_width=2.0, border_width=1.0, holes=False),
        "steps_up": T.MeshInvertedPyramidStairsTerrainCfg(proportion=1.0, step_height_range=STEP_RANGE, step_width=0.4, platform_width=2.0, border_width=1.0, holes=False)}
for k, a in enumerate(ROUGH_AMPS):
    subs[f"rough_{k}"] = T.HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.0, a), noise_step=0.01, downsampled_scale=0.2, border_width=0.25)
names = list(subs.keys()); COL = {n: i for i, n in enumerate(names)}
cells = [("flat", 0)] + [(t, r) for t in ("slope_down", "slope_up", "steps_down", "steps_up") for r in range(ROWS)] + [(f"rough_{k}", 0) for k in range(len(ROUGH_AMPS))]
if xa.only: cells = [(c.split(':')[0], int(c.split(':')[1])) for c in xa.only.split(',')]
N = len(cells) * xa.reps
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.scene.height_scanner = None; env_cfg.observations.policy.height_scan = None; env_cfg.events.push_robot = None; env_cfg.curriculum.terrain_levels = None
cm = env_cfg.commands.base_velocity; cm.heading_command = False; cm.rel_standing_envs = 0.0; cm.resampling_time_range = (1e6, 1e6); cm.debug_vis = False
cm.ranges.lin_vel_x = (xa.fwd, xa.fwd); cm.ranges.lin_vel_y = (0.0, 0.0); cm.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(size=(20.0, 20.0), border_width=20.0, num_rows=ROWS, num_cols=len(names), horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
                                                              use_cache=False, curriculum=True, difficulty_range=(0.0, 1.0), sub_terrains=subs)
env_cfg.scene.terrain.max_init_terrain_level = None; env_cfg.scene.num_envs = N; args.num_envs = N
if xa.record_dir:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    S_ = int(round(xa.horizon_s / 0.02))
    # approximate centroid of the selected sub-terrains (grid of 20 m patches centred on the origin); printed origins in the log allow checking it
    _cx = float(np.mean([(r_ + 0.5) * 20.0 - ROWS * 20.0 / 2 for (t_, r_) in cells])); _cy = float(np.mean([(COL[t_] + 0.5) * 20.0 - len(names) * 20.0 / 2 for (t_, r_) in cells])); _ex, _ey, _ez = [float(v) for v in xa.cam.split(",")]
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, eye=(_cx + _ex, _cy + _ey, _ez), lookat=(_cx, _cy, 0.0))]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.record_dir, video_length=S_, video_interval=S_, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = xa.seed; env_cfg.episode_length_s = xa.horizon_s + 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; assert u.num_envs == N, (u.num_envs, N); r = u.scene["robot"]
    wcfg = handle_deprecated_rsl_rl_cfg(G1FlatPPORunnerCfg(), check_rsl_rl_version()); wcfg.device = agent_cfg.device
    wr = create_rsl_rl_runner(env, wcfg); wr.load(xa.walk_checkpoint, map_location=agent_cfg.device); pol = wr.get_inference_policy(device=dev)
    terr = u.scene.terrain; rows_t = torch.tensor([r_ for (t_, r_) in cells for _ in range(xa.reps)], device=dev); cols_t = torch.tensor([COL[t_] for (t_, r_) in cells for _ in range(xa.reps)], device=dev)
    terr.terrain_levels[:] = rows_t; terr.terrain_types[:] = cols_t; terr.env_origins[:] = terr.terrain_origins[rows_t, cols_t]
    print('TERRAIN_ORIGINS_SAMPLE', terr.terrain_origins[0, 0].tolist(), terr.terrain_origins[ROWS - 1, len(names) - 1].tolist(), flush=True)
    obs, _ = env.reset(); term = u.command_manager.get_term("base_velocity")
    STEPS = int(round(xa.horizon_s / 0.02)); org = u.scene.env_origins
    q0 = r.data.root_quat_w.torch.clone(); head = torch.stack([1 - 2 * (q0[:, 1] ** 2 + q0[:, 2] ** 2), 2 * (q0[:, 0] * q0[:, 1] + q0[:, 3] * q0[:, 2])], -1)
    p0 = (r.data.root_pos_w.torch[:, :2] - org[:, :2]).clone(); last = p0.clone(); dead = torch.zeros(N, dtype=torch.bool, device=dev); dstep = torch.full((N,), -1, device=dev, dtype=torch.long)
    pend = p0.clone(); h_min = torch.full((N,), 9.0, device=dev); resets_seen = 0
    for k in range(STEPS):
        term.vel_command_b[:] = 0.0; term.vel_command_b[:, 0] = xa.fwd
        with torch.inference_mode():
            obs, _, dones, _ = env.step(pol(obs))
        newly = dones.bool() & ~dead; dstep = torch.where(newly, torch.full_like(dstep, k), dstep); pend = torch.where(newly[:, None], last, pend); dead |= newly
        cur = r.data.root_pos_w.torch[:, :2] - org[:, :2]; last = torch.where(dead[:, None], last, cur)
    pend = torch.where(dead[:, None], pend, last)
    along, lateral = M.heading_progress(p0.cpu(), pend.cpu(), head.cpu()); surv = ~dead.cpu(); ds = dstep.cpu()
    rows_c = rows_t.cpu(); cols_c = cols_t.cpu()
    by = {}
    for (t_, r_) in cells:
        sel = (rows_c == r_) & (cols_c == COL[t_]); k_ = int((surv & sel).sum()); n_ = int(sel.sum()); lo, hi = RS.wilson_ci(k_, n_)
        a_ = along[sel & surv]
        by.setdefault(t_, {})[r_] = dict(n=n_, survived=k_, rate=k_ / n_, wilson95=[round(lo, 3), round(hi, 3)], along_m_survivors_mean=None if len(a_) == 0 else round(float(a_.mean()), 3),
                                         speed_m_s_survivors_mean=None if len(a_) == 0 else round(float(a_.mean()) / xa.horizon_s, 3), along_m_all_mean=round(float(along[sel].mean()), 3),
                                         fall_time_s_mean=None if int((~surv & sel).sum()) == 0 else round(float((ds[sel & ~surv].float() * 0.02).mean()), 2))
    desc = dict(rows=ROWS, slope_max_ratio=SLOPE_MAX, slope_deg_by_level=[round(math.degrees(math.atan(SLOPE_MAX * (r_ + 0.5) / ROWS)), 1) for r_ in range(ROWS)],
                step_height_m_by_level=[round(STEP_RANGE[0] + (STEP_RANGE[1] - STEP_RANGE[0]) * (r_ + 0.5) / ROWS, 3) for r_ in range(ROWS)], rough_noise_amplitude_m=ROUGH_AMPS, step_width_m=0.4)
    res = dict(protocol="M3 terrain baseline (evaluation only)", task=args.task, walk_checkpoint=xa.walk_checkpoint, seed=xa.seed, num_envs=N, reps_per_cell=xa.reps, horizon_s=xa.horizon_s, fwd_cmd=xa.fwd,
               observation="123-d, height scan removed (blind walking); pushes off", level_descriptors=desc, by_type_level=by, overall=dict(n=N, survived=int(surv.sum())),
               per_env=dict(type=[cells[i // xa.reps][0] for i in range(N)], level=[cells[i // xa.reps][1] for i in range(N)], survived=surv.int().tolist(), along_m=[round(float(a_), 3) for a_ in along], fall_step=ds.tolist()))
    json.dump(res, open(xa.out, "w"), indent=1); print("M3_JSON", json.dumps(dict(overall=res["overall"], flat=by.get("flat", {}).get(0))), flush=True)
    C.close_env(env)
