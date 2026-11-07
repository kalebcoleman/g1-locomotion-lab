"""Chain evaluation: fall -> get up -> stand -> walk, on the held-out fallen poses (full-collision G1, Newton MJWarp).
Recovery policy runs until the env has held the standing criterion for HOLD_S (the same success definition), then the EXISTING walking
policy (Isaac-Velocity-Flat-G1, iteration 1499 of the published run, trained on the minimal-collision G1) takes over with a forward command.
  python eval_chain.py --task G1-Recovery-A-v0 --checkpoint <recovery model.pt> --walk_checkpoint <walking model.pt> --num_envs 256 --out r.json [--record_dir D --label L --num_envs 16] physics=newton_mjwarp
Reported: chain stages reached, walking metrics after hand-over, falls after hand-over. The recovery-only success is evaluated separately by eval_recovery.py."""
import sys, os, json, argparse, time
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out", required=True); ex.add_argument("--walk_checkpoint", required=True)
ex.add_argument("--handover_steps", type=int, default=None, help="consecutive standing steps (pelvis>=0.60 m, tilt<=26deg) before the walking policy takes over; default = the full 2 s hold (100 steps)")
ex.add_argument("--fwd", type=float, default=0.8); ex.add_argument("--total_s", type=float, default=20.0)
ex.add_argument("--pose_indices", default=None, help="comma list of held-out pose indices to use for env 0..N-1 (e.g. to film specific attempts; labelled as selected)")
ex.add_argument("--record_dir", default=None); ex.add_argument("--label", default="chain")
ex.add_argument("--heldout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heldout_poses.npz"))
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
if xa.record_dir: args.enable_cameras = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = os.path.abspath(xa.heldout)
if xa.record_dir:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    CYC = int(xa.total_s / 0.02)
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, eye=(0, -11, 9), lookat=(0, 0, 0.3))]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.record_dir, video_length=CYC, video_interval=CYC, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1; env_cfg.episode_length_s = xa.total_s + 10.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs
    rec_runner = create_rsl_rl_runner(env, agent_cfg); rec_runner.load(args.checkpoint, map_location=agent_cfg.device)
    rec_pol = rec_runner.get_inference_policy(device=dev)
    wcfg = handle_deprecated_rsl_rl_cfg(G1FlatPPORunnerCfg(), check_rsl_rl_version()); wcfg.device = agent_cfg.device
    walk_runner = create_rsl_rl_runner(env, wcfg); walk_runner.load(xa.walk_checkpoint, map_location=agent_cfg.device)
    walk_pol = walk_runner.get_inference_policy(device=dev)
    f = np.load(xa.heldout); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    if xa.pose_indices:
        pi = torch.tensor([int(v) for v in xa.pose_indices.split(",")], device=dev)
        u._rec_fixed = {k: v[pi] for k, v in u._rec_fixed.items()}
    obs, _ = env.reset(); t = K.get_tracker(u); r = u.scene["robot"]
    term = u.command_manager.get_term("base_velocity")
    cat = u._rec_fixed["category"][torch.arange(N, device=dev) % u._rec_fixed["z"].shape[0]]
    steps = int(round(xa.total_s / u.step_dt)); HO = xa.handover_steps if xa.handover_steps is not None else t.hold_steps_needed
    switched = torch.zeros(N, dtype=torch.bool, device=dev); sw_step = torch.full((N,), -1, device=dev, dtype=torch.long)
    p0 = torch.zeros(N, 2, device=dev); fell_after = torch.zeros(N, dtype=torch.bool, device=dev); minh_after = torch.full((N,), 9.0, device=dev)
    vx_sum = torch.zeros(N, device=dev); n_after = torch.zeros(N, device=dev)
    for s in range(steps):
        sw_now = (t.run >= HO) & ~switched
        if sw_now.any():
            p0[sw_now] = (r.data.root_pos_w.torch[sw_now, :2] - u.scene.env_origins[sw_now, :2]); sw_step[sw_now] = s
        switched |= sw_now
        term.vel_command_b[:] = 0.0
        term.vel_command_b[switched, 0] = xa.fwd
        with torch.inference_mode():
            a_rec = rec_pol(obs); a_walk = walk_pol(obs)
            obs, _, _, _ = env.step(torch.where(switched[:, None], a_walk, a_rec))
        h, up = (r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2], -r.data.projected_gravity_b.torch[:, 2])
        minh_after = torch.where(switched, torch.minimum(minh_after, h), minh_after)
        fell_after |= switched & ((h < 0.45) | (up < 0.5))
        vx_sum += torch.where(switched, r.data.root_lin_vel_b.torch[:, 0], torch.zeros_like(h)); n_after += switched.float()
    h, up = (r.data.root_pos_w.torch[:, 2] - u.scene.env_origins[:, 2], -r.data.projected_gravity_b.torch[:, 2])
    disp = torch.linalg.norm((r.data.root_pos_w.torch[:, :2] - u.scene.env_origins[:, :2]) - p0, dim=1)
    standing_end = (h >= K.SUCCESS_H) & (up >= K.SUCCESS_TILT_COS)
    walked = switched & ~fell_after & standing_end & (disp >= 1.0)
    dt = u.step_dt; fc = t.failure_cause()
    sel_idx = (pi.cpu().tolist() if xa.pose_indices else list(range(N)))
    res = dict(success_pose_indices=[sel_idx[i] for i in torch.nonzero(walked).squeeze(-1).tolist()], recovered_pose_indices=[sel_idx[i] for i in torch.nonzero(t.succ).squeeze(-1).tolist()], handover_steps=int(HO), handover_rule="consecutive standing steps before the walking policy takes over (no reset/teleport; same physics state)", task=args.task, checkpoint=args.checkpoint, walk_checkpoint=xa.walk_checkpoint, num_envs=N, total_s=xa.total_s, fwd_cmd=xa.fwd,
               stage1_recovered=int(t.succ.sum()), stage1_rate=float(t.succ.float().mean()),
               stage2_handover_then_no_fall=int((switched & ~fell_after).sum()),
               stage3_chain_success=int(walked.sum()), chain_success_rate=float(walked.float().mean()),
               fell_after_handover=int((switched & fell_after).sum()),
               handover_time_s_mean=float((sw_step[switched].float() * dt).mean()) if switched.any() else None,
               walk_displacement_m_mean=float(disp[switched].mean()) if switched.any() else None,
               walk_fwd_speed_mean=float((vx_sum[switched] / n_after[switched].clamp(min=1)).mean()) if switched.any() else None,
               recovery_failure_causes={n: int((fc == i).sum()) for i, n in enumerate(K.Tracker.FAIL)},
               chain_by_category={c: dict(n=int((cat == i).sum()), chain_success_rate=float(walked[cat == i].float().mean())) for i, c in enumerate(K.CATEGORIES)},
               note="walking policy trained on the minimal-collision G1; here applied to the full-collision G1 (same joints/actuators)")
    json.dump(res, open(xa.out, "w"), indent=1); print("CHAIN_JSON", json.dumps(res), flush=True)
    C.close_env(env)
