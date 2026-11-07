"""Evaluate a recovery checkpoint on the HELD-OUT fallen poses (256 poses, 4 fall directions, one 10 s episode each).
  python eval_recovery.py --task G1-Recovery-A-v0 --checkpoint <model.pt> --num_envs 256 --out result.json [--video_dir D] physics=newton_mjwarp
Metrics: recovery success (stand >= 2 s: pelvis >= 0.60 m and torso tilt <= 26 deg), recovery time (start of the successful hold),
longest/total standing time, joint-limit violations (soft limits), failure causes. Needs the held-out npz (made on first use)."""
import sys, os, json, time, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out", required=True)
ex.add_argument("--heldout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heldout_poses.npz"))
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest)
args.train_env_cfg = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = os.path.abspath(xa.heldout)   # base cfg passes this to reset_fallen
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1
    env_cfg.episode_length_s = 20.0     # we stop after exactly 10 s so the tracker is never reset mid-read
    if not os.path.exists(xa.heldout):
        pass
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs
    r = u.scene["robot"]
    if not os.path.exists(xa.heldout):
        dq = r.data.default_joint_pos.torch[0]; lim = r.data.soft_joint_pos_limits.torch[0]
        mask = torch.tensor([0.0 if K.is_hand_joint(n) else 1.0 for n in r.joint_names], device=dev)
        K.make_heldout(dq, lim[:, 0], lim[:, 1], mask, path=xa.heldout); print("made held-out set", xa.heldout)
    runner = create_rsl_rl_runner(env, agent_cfg)
    runner.load(args.checkpoint, map_location=agent_cfg.device) if args.checkpoint else None
    policy = runner.get_inference_policy(device=dev)
    # re-reset so the (possibly just created) held-out file is used
    u._rec_fixed = None
    f = np.load(xa.heldout); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    obs, _ = env.reset()
    steps = int(round(K.EPISODE_S / u.step_dt)); t = K.get_tracker(u)
    cat = u._rec_fixed["category"][torch.arange(N, device=dev) % u._rec_fixed["z"].shape[0]]
    t0 = time.time(); hmin_trace = []
    for s in range(steps):
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
    wall = time.time() - t0
    dt = u.step_dt; ok = t.succ
    res = dict(task=args.task, checkpoint=args.checkpoint, num_envs=N, episode_s=K.EPISODE_S, steps=steps, success_hold_s=K.HOLD_S,
               success_rate=float(ok.float().mean()), successes=int(ok.sum()),
               recovery_time_s_mean=float((t.succ_start[ok].float() * dt).mean()) if ok.any() else None,
               recovery_time_s_median=float((t.succ_start[ok].float() * dt).median()) if ok.any() else None,
               longest_stand_s_mean=float((t.best_run.float() * dt).mean()), total_stand_s_mean=float((t.total_stand.float() * dt).mean()),
               joint_limit_violation_step_frac=float(t.viol_steps.float().mean() / steps),
               joint_limit_violations_per_step_mean=float(t.viol_joint_steps.float().mean() / steps), eval_wall_s=wall)
    ok_s = t.succ_s
    res["strict_supported_stand"] = dict(definition="pelvis>=0.60 m, torso tilt<=26 deg, BOTH feet contact force>10 N and feet<0.15 m above floor, pelvis speed<0.6 m/s, held 2 s",
        success_rate=float(ok_s.float().mean()), successes=int(ok_s.sum()), recovery_time_s_mean=float((t.succ_start_s[ok_s].float() * dt).mean()) if ok_s.any() else None,
        longest_stand_s_mean=float((t.best_run_s.float() * dt).mean()), total_stand_s_mean=float((t.total_stand_s.float() * dt).mean()),
        envs_ever_supported_stand=int(t.ever_stood_s.sum()), by_category={CAT: float(ok_s[cat == i].float().mean()) for i, CAT in enumerate(K.CATEGORIES)})
    vvf = t.vel_viol_steps.float() / steps; plaus = t.succ_s & (vvf <= 0.02) & (t.peak_vz <= 2.0)
    res["plausibility"] = dict(definition="PLAUSIBLE recovery = strict supported stand held 2 s AND joint-velocity-limit violations on <=2% of steps AND peak pelvis upward speed <=2.0 m/s",
        plausible_successes=int(plaus.sum()), plausible_rate=float(plaus.float().mean()), vel_limit_violation_step_frac_mean=float(vvf.mean()), peak_pelvis_vz_mean=float(t.peak_vz.mean()), peak_pelvis_vz_max=float(t.peak_vz.max()),
        strict_successes_failing_plausibility=int((t.succ_s & ~plaus).sum()),
        flight_step_frac_mean=float((t.flight_steps.float() / steps).mean()),
        plausible_v1p1_successes=int((plaus & (t.flight_steps.float() / steps <= 0.02)).sum()), v1p1_definition="POST-HOC amendment (added after seeing A8c): plausible AND free-flight (no body in contact with anything) on <=2% of steps")
    fc = t.failure_cause()
    res["failure_causes"] = {n: int((fc == i).sum()) for i, n in enumerate(K.Tracker.FAIL)}
    res["by_category"] = {CAT: dict(n=int((cat == i).sum()), success_rate=float(ok[cat == i].float().mean())) for i, CAT in enumerate(K.CATEGORIES)}
    if hasattr(u, "_assist_log"):
        L = u._assist_log; af = L["force_steps"]
        res["assist"] = dict(level_env_var=os.environ.get("G1R_EVAL_ASSIST"), envs_with_force=int((af > 0).sum()), force_steps_mean=float(af.mean()), force_seconds_mean=float(af.mean() * u.step_dt),
                             impulse_Ns_mean=float(L["impulse"].mean()), max_force_N=float(L["max_force"].max()), first_step_mean=float(L["first"][af > 0].mean()) if (af > 0).any() else None,
                             last_step_mean=float(L["last"][af > 0].mean()) if (af > 0).any() else None, robot_weight_N=getattr(u, "_robot_weight", None),
                             note="force applies only while torso upright (gravity z < -0.8) and pelvis < 0.62 m; unassisted eval = G1R_EVAL_ASSIST unset or 0")
    json.dump(res, open(xa.out, "w"), indent=1)
    print("EVAL_JSON", json.dumps(res), flush=True)
    C.close_env(env)
