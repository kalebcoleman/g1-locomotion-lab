"""M2 (EVALUATION ONLY, no training): one continuous episode  walk -> induced fall -> get-up -> stand gate -> walk, both policies frozen.
  python eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint <recovery.pt> --walk_checkpoint <walking.pt> --num_envs 64 --seed 1 --out r.json \
         [--levels 0,1.5,2.5,3.5] [--record_dir D --label L] physics=newton_mjwarp
Timeline per env (control step 0.02 s): start standing; frozen WALKING policy walks at a commanded forward velocity; at --push_s a horizontal root-velocity impulse of the env's
magnitude (m/s, random direction, fixed by --seed) is ADDED to the base velocity (same mechanism as Isaac Lab's push_by_setting_velocity, but with a logged per-env vector).
Magnitude 0 is the no-push control. A fall is detected when pelvis < 0.45 m or torso up < 0.5 for --det_steps consecutive steps; the frozen RECOVERY policy then acts with a zero
command until the criteria-v3 stand gate passes over the get-up window (fall_handover.window_gate); then one-way hand-over to the walking policy for --walk_s seconds.
No env reset is ever performed or credited: a reset (done flag) anywhere before the walking window completes is a failure (F_RESET).
Observation/action compatibility: both policies read the SAME 123-d observation from the SAME env (base lin vel 3, base ang vel 3, projected gravity 3, velocity command 3, joint pos 37,
joint vel 37, last action 37; no observation normalizer in either checkpoint) and emit 37-d joint-position targets through the same action term (scale 0.5, default offset, no clip).
Physics note: the walking policy was trained on the minimal-collision G1; here it runs on the full-collision G1 with self-collision and Unitree's real torque limits (the R9hg env).
Pose sets are not used (the fall comes from the dynamics), so DEV/HELDOUT/TEST are not touched; results are labelled DEV by convention (seeds 1..3)."""
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
ex.add_argument("--levels", default="0,1.5,2.5,3.5", help="push magnitudes (m/s of added horizontal base velocity), cycled over envs; 0 = no-push control")
ex.add_argument("--seed", type=int, default=1); ex.add_argument("--fwd", type=float, default=0.8)
ex.add_argument("--push_s", type=float, default=3.0); ex.add_argument("--nofall_s", type=float, default=3.0)
ex.add_argument("--det_steps", type=int, default=10); ex.add_argument("--recover_max_s", type=float, default=10.0); ex.add_argument("--walk_s", type=float, default=10.0)
ex.add_argument("--total_s", type=float, default=27.0)
ex.add_argument("--gate", choices=["v3", "strict"], default="v3", help="v3 = criteria v3 stand/hold over the get-up window (headline); strict = supported 2 s stand only (EXPLORATORY, labelled as such)")
ex.add_argument("--trigger", choices=["v3", "ready"], default="v3", help="M2b: what fires the hand-over. v3 = official window gate (M2 behaviour); ready = separate hand-over READINESS check (supported stance, speed < 0.2 m/s, angular speed < 0.5 rad/s, held --ready_hold steps); the official v3 result is still tracked")
ex.add_argument("--ready_hold", type=int, default=50); ex.add_argument("--settle", type=int, default=0, help="steps between the trigger and the switch (recovery policy or hold keeps acting)")
ex.add_argument("--settle_mode", choices=["recovery", "hold"], default="recovery", help="hold = command the current joint positions during the settle"); ex.add_argument("--blend", type=int, default=0, help="steps (<= settle) at the end of the settle over which the action ramps to the default stance")
ex.add_argument("--blend_joints", choices=["all", "hands", "upper"], default="all", help="which joints the blend drives to the default stance: all, hands (finger joints only), upper (shoulders, elbows and fingers; legs and torso untouched)")
ex.add_argument("--ramp", type=int, default=0, help="steps over which the commanded forward velocity ramps from 0 to --fwd after the switch")
ex.add_argument("--record_dir", default=None); ex.add_argument("--label", default="m2")
# recording camera only (defaults = the M2/M2b camera, unchanged). --eye/--lookat are the fixed world-space camera position and target.
ex.add_argument("--eye", default="0,-11,9"); ex.add_argument("--lookat", default="0,0,0.3"); ex.add_argument("--cam_focal", type=float, default=12.0)
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
if xa.record_dir: args.enable_cameras = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K, metrics as M, fall_handover as FH
from g1_recovery.lane_a import free_flight_ground
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
DT = 0.02; STEPS = int(round(xa.total_s / DT))
if xa.record_dir:
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    env_cfg.sim.visualizer_cfgs = [KitVisualizerCfg(headless=True, window_width=1280, window_height=720, focal_length=xa.cam_focal,
                                                     eye=tuple(float(v) for v in xa.eye.split(",")), lookat=tuple(float(v) for v in xa.lookat.split(",")))]
    env_cfg.video_recorders = [VideoRecorderCfg(source="visualizer:kit", output_dir=xa.record_dir, video_length=STEPS, video_interval=STEPS, output_filename_prefix=xa.label)]
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = xa.seed; env_cfg.episode_length_s = xa.total_s + 30.0       # the env's own time-out must never fire inside the horizon
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]
    assert abs(u.step_dt - DT) < 1e-6, u.step_dt
    rec_runner = create_rsl_rl_runner(env, agent_cfg); rec_runner.load(args.checkpoint, map_location=agent_cfg.device); rec_pol = rec_runner.get_inference_policy(device=dev)
    wcfg = handle_deprecated_rsl_rl_cfg(G1FlatPPORunnerCfg(), check_rsl_rl_version()); wcfg.device = agent_cfg.device
    walk_runner = create_rsl_rl_runner(env, wcfg); walk_runner.load(xa.walk_checkpoint, map_location=agent_cfg.device); walk_pol = walk_runner.get_inference_policy(device=dev)
    # start every env STANDING (default pose) instead of lying: reset_fallen reads env._rec_fixed
    defq = r.data.default_joint_pos.torch[0].clone()
    u._rec_fixed = dict(z=torch.tensor([0.74], device=dev), quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=dev), q=defq[None].clone(), category=torch.zeros(1, device=dev, dtype=torch.long))
    obs, _ = env.reset(); t = K.get_tracker(u); term = u.command_manager.get_term("base_velocity")
    HOLD = t.hold_steps_needed; PUSH = int(round(xa.push_s / DT)); WALK_STEPS = int(round(xa.walk_s / DT))
    fsm = FH.FallHandoverFSM(N, dev, det_steps=xa.det_steps, recover_max=int(round(xa.recover_max_s / DT)), walk_steps=WALK_STEPS, nofall_steps=int(round(xa.nofall_s / DT)))
    hc = FH.HandoverController(N, dev, trigger=xa.trigger, ready_hold=xa.ready_hold, settle=xa.settle, blend=xa.blend, ramp=xa.ramp); ACT_SCALE = float(env_cfg.actions.joint_pos.scale)
    jgroup = lambda n_: "finger" if K.is_hand_joint(n_) else "arm" if ("shoulder" in n_ or "elbow" in n_) else "torso" if "torso" in n_ else "leg"
    bmask = torch.tensor([1.0 if (xa.blend_joints == "all" or (xa.blend_joints == "hands" and jgroup(n_) == "finger") or (xa.blend_joints == "upper" and jgroup(n_) in ("finger", "arm"))) else 0.0 for n_ in r.joint_names], device=dev)
    levels = [float(v) for v in xa.levels.split(",")]; mag = torch.tensor([levels[i % len(levels)] for i in range(N)], device=dev)
    g = torch.Generator().manual_seed(xa.seed * 1000 + 7); ang = (torch.rand(N, generator=g) * 2 * math.pi).to(dev)
    _lim = torch.tensor([M.real_torque_limit(n) for n in r.joint_names], device=dev); _bj = torch.isfinite(_lim); _bji = torch.nonzero(_bj).squeeze(-1)
    _fb = [r.body_names.index(n) for n in r.body_names if "ankle_roll" in n]
    _hip = [r.joint_names.index(n) for n in ("left_hip_roll_joint", "right_hip_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint")]
    org = u.scene.env_origins
    z32 = lambda: torch.zeros(N, device=dev, dtype=torch.int32)
    wsteps, air_g, tq_over = z32(), z32(), z32(); strict_step = torch.full((N,), -1, device=dev, dtype=torch.long)
    sbuf = torch.zeros(HOLD, N, device=dev)
    TR = {k: [] for k in ["h", "up", "stance", "hip", "jv", "act", "pos", "quat", "state", "obs"]}
    otensor = lambda o: o["policy"] if hasattr(o, "keys") and "policy" in o.keys() else o
    n_reset_events = 0; ND = {}
    for k in range(STEPS):
        if k == PUSH:
            m = fsm.mark_impulse(k, mag > 0)
            if m.any():
                ids = torch.nonzero(m).squeeze(-1); vel = r.data.root_vel_w.torch[ids].clone()
                vel[:, 0] += mag[ids] * torch.cos(ang[ids]); vel[:, 1] += mag[ids] * torch.sin(ang[ids])
                r.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=ids)
        walkmask = (fsm.state == FH.WALK) | (fsm.state == FH.WALK2)
        in_rec0 = fsm.state == FH.RECOVER; in_w2 = fsm.state == FH.WALK2
        term.vel_command_b[:] = 0.0; term.vel_command_b[walkmask, 0] = xa.fwd * hc.command_scale(k - 1, fsm.handover_step, in_w2)[walkmask]
        with torch.inference_mode():
            a_rec = rec_pol(obs); a_walk = walk_pol(obs)
            base = a_rec
            if xa.settle_mode == "hold":                                       # hold the current joint positions while settling
                base = torch.where(hc.settling(in_rec0)[:, None], (r.data.joint_pos.torch - defq[None]) / ACT_SCALE, a_rec)
            al = hc.blend_alpha(k - 1, in_rec0)[:, None] * bmask[None]; base = base * (1.0 - al)   # blend toward the default stance (action 0) at the end of the settle
            act = torch.where(walkmask[:, None], a_walk, base)
            obs, _, dones, _ = env.step(act)
        h = r.data.root_pos_w.torch[:, 2] - org[:, 2]; up = -r.data.projected_gravity_b.torch[:, 2]
        pf = r.data.body_pos_w.torch[:, _fb, :2]; stance = torch.linalg.norm(pf[:, 0] - pf[:, 1], dim=-1); sbuf[k % HOLD] = stance
        rec_now = fsm.state == FH.RECOVER                                   # state before this step's transition
        if rec_now.any():
            wsteps += rec_now.int()
            air_g += (rec_now & (free_flight_ground(u) > 0.5)).int()
            tq_over += (rec_now & (((r.data.applied_torque.torch.abs() / _lim) > 1.0) & _bj).any(-1)).int()
            strict_step = torch.where(rec_now & (t.run_s >= HOLD) & (strict_step < 0), torch.full_like(strict_step, k), strict_step)
        v3_ok = (FH.window_gate(t.run_s, HOLD, wsteps, t.vel_viol_steps, air_g, t.viol_steps, t.peak_vz, t.hard_viol_steps, t.hard_excess_max, tq_over, sbuf.mean(0), h, up) if xa.gate == "v3"
                 else FH.strict_gate(t.run_s, HOLD, h, up)) & rec_now & (wsteps >= HOLD)
        sup = K.supported_mask(u); spd = torch.linalg.norm(r.data.root_lin_vel_w.torch, dim=-1); angs = torch.linalg.norm(r.data.root_ang_vel_w.torch, dim=-1)
        gate = hc.update(k, rec_now, v3_ok, FH.ready_check(h, up, sup, spd, angs), sup & (h >= M.P.stand_h) & (up >= M.P.stand_up))
        reset = dones.bool(); n_reset_events += int(reset.sum())
        entered = fsm.step(k, h, up, gate, reset)
        tmo = (fsm.state == FH.F_NOSTAND) & rec_now                       # recovery window just timed out: record which gate clause was missing
        if tmo.any():
            ids = torch.nonzero(tmo).squeeze(-1); w_ = wsteps[ids].clamp(min=1).float()
            for key, val in dict(best_run_steps=t.best_run_s[ids].float(), vel_frac=t.vel_viol_steps[ids] / w_, flight_frac=air_g[ids] / w_, pos_frac=t.viol_steps[ids] / w_, hard_frac=t.hard_viol_steps[ids] / w_,
                                 hard_excess=t.hard_excess_max[ids], torque_frac=tq_over[ids] / w_, peak_vz=t.peak_vz[ids], stance_hold=sbuf.mean(0)[ids], h=h[ids], up=up[ids]).items():
                ND.setdefault(key, torch.full((N,), float("nan"), device=dev))[ids] = val.float()
        if entered.any():
            ids = torch.nonzero(entered).squeeze(-1)
            with torch.inference_mode(): t.reset(ids)                    # tracker tensors are inference tensors
            wsteps[ids] = 0; air_g[ids] = 0; tq_over[ids] = 0     # tracker counters restart at fall detection (bookkeeping, not an env reset)
        TR["h"].append(h); TR["up"].append(up); TR["stance"].append(stance); TR["hip"].append(r.data.joint_pos.torch[:, _hip]); TR["jv"].append(r.data.joint_vel.torch[:, _bji])
        TR["act"].append(act.clone()); TR["pos"].append(r.data.root_pos_w.torch[:, :2] - org[:, :2]); TR["quat"].append(r.data.root_quat_w.torch.clone()); TR["state"].append(fsm.state.clone()); TR["obs"].append(otensor(obs).clone())
    fsm.finalize(STEPS)
    TR = {k: torch.stack(v, 0) for k, v in TR.items()}
    S = fsm.state.cpu(); imp = fsm.impulse_step.cpu(); det = fsm.detect_step.cpu(); hov = fsm.handover_step.cpu(); endk = fsm.end_step.cpu(); onset = fsm.fall_onset.cpu()
    impulsed = fsm.impulsed.cpu(); spont = fsm.spontaneous.cpu(); mag_c = mag.cpu(); strict_c = strict_step.cpu()
    entered_rec = det >= 0; induced = impulsed & entered_rec & ~spont
    handed = hov >= 0; done = S == FH.DONE; refell = S == FH.F_REFALL; resetf = S == FH.F_RESET
    # post-handover walking: along-heading displacement over the walking window (or to the end if it ended early)
    p0 = torch.zeros(N, 2); p1 = torch.zeros(N, 2); head = torch.zeros(N, 2); pos = TR["pos"].cpu(); quat = TR["quat"].cpu()
    for i in torch.nonzero(handed).squeeze(-1).tolist():
        s0 = int(hov[i]); s1 = min(s0 + WALK_STEPS, STEPS - 1) if done[i] else int(endk[i]) if int(endk[i]) >= 0 else STEPS - 1
        q = quat[s0, i]; head[i] = torch.tensor([1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[3] * q[2])]); p0[i] = pos[s0, i]; p1[i] = pos[min(s1, STEPS - 1), i]
    along, lateral = M.heading_progress(p0, p1, head)
    chain_ok, reason = M.chain_success(handed, handed, done | refell, refell, torch.where(handed, along, torch.zeros_like(along)), xa.walk_s)      # a window ended by a re-fall is complete-with-fall (reason fell_after_handover)
    success = chain_ok & induced & ~resetf
    # reference: the same walking policy in the no-push control, along-heading distance over an identical window (from the push time, after the same 3 s walk-in)
    ctl = (mag_c == 0) & (S == FH.WALK); cp0 = torch.zeros(N, 2); cp1 = torch.zeros(N, 2); chd = torch.zeros(N, 2)
    for i in torch.nonzero(ctl).squeeze(-1).tolist():
        q = quat[PUSH, i]; chd[i] = torch.tensor([1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[3] * q[2])]); cp0[i] = pos[PUSH, i]; cp1[i] = pos[min(PUSH + WALK_STEPS, STEPS - 1), i]
    c_along, _ = M.heading_progress(cp0, cp1, chd); c_along = c_along[ctl]
    getup_s = torch.where(handed, (hov - det).float() * DT, torch.full((N,), float("nan"))); strict_s = torch.where(strict_c >= 0, (strict_c - det).float() * DT, torch.full((N,), float("nan")))
    lat_s = torch.where(induced, (det - imp).float() * DT, torch.full((N,), float("nan")))
    walk_time = torch.where(handed, torch.where(done, torch.full((N,), xa.walk_s), (endk - hov).float().clamp(min=0) * DT), torch.full((N,), float("nan")))
    # M1 movement metrics over each env's get-up window [detect, hand-over) (whole recovery window if no hand-over)
    mv = {}
    for i in torch.nonzero(entered_rec).squeeze(-1).tolist():
        a = int(det[i]); b = int(hov[i]) if int(hov[i]) >= 0 else min(a + int(round(xa.recover_max_s / DT)), STEPS)
        if b - a < 60: continue
        sl = slice(a, b); col = slice(i, i + 1)
        m_ = M.movement_metrics(TR["stance"][sl, col], TR["h"][sl, col], TR["hip"][sl, col], TR["jv"][sl, col], TR["act"][sl, col], DT)
        for kk, vv in m_.items(): mv.setdefault(kk, []).append(float(vv[0]))
    # ---- M2b extras ----------------------------------------------------------------------------------------------------------
    v3_first = hc.v3_first.cpu(); v3_at = hc.v3_at_handover.cpu() & handed; trig = hc.trig_step.cpu(); ready_first = hc.ready_first.cpu(); cancels = hc.cancels.cpu()
    ok3 = done & handed & (along >= M.P.walk_dist) & induced & ~resetf                      # weaker walking criterion: 10 s without a re-fall and >= 3.0 m (no speed rule)
    ttr = torch.where(refell & handed, (endk - hov).float() * DT, torch.full((N,), float("nan")))
    # walking from rest in the no-push control (start-up included): along-heading distance over the first walk_s seconds, for a fair speed reference
    cr0 = torch.zeros(N, 2); cr1 = torch.zeros(N, 2); crh = torch.zeros(N, 2)
    for i in torch.nonzero(ctl).squeeze(-1).tolist():
        q = quat[0, i]; crh[i] = torch.tensor([1 - 2 * (q[1] ** 2 + q[2] ** 2), 2 * (q[0] * q[1] + q[3] * q[2])]); cr0[i] = pos[0, i]; cr1[i] = pos[min(WALK_STEPS, STEPS - 1), i]
    r_along, _ = M.heading_progress(cr0, cr1, crh); r_along = r_along[ctl]
    # diagnosis: policy-input observation at the hand-over vs the walking policy's usual distribution (no-push control, the 2 s before the push)
    OBS = TR["obs"].cpu(); diag = {}
    cidx = torch.nonzero(ctl).squeeze(-1)
    if len(cidx) >= 4 and handed.any():
        ref = OBS[PUSH - 100:PUSH - 1][:, cidx].reshape(-1, OBS.shape[-1]); mu = ref.mean(0); sd = ref.std(0).clamp(min=0.02)      # std floor 0.02 so near-constant inputs (finger joints) do not give meaningless z
        jn_ = list(r.joint_names); nm = [f"base_lin_vel_{a}" for a in "xyz"] + [f"base_ang_vel_{a}" for a in "xyz"] + [f"proj_gravity_{a}" for a in "xyz"] + [f"cmd_{a}" for a in "xyz"] + [f"q:{j}" for j in jn_] + [f"qd:{j}" for j in jn_] + [f"a:{j}" for j in jn_]
        hm = torch.tensor([K.is_hand_joint(j) for j in jn_]); nj = len(jn_); L = lambda off, m: torch.nonzero(m).squeeze(-1) + off
        terms = dict(base_lin_vel=torch.arange(0, 3), base_ang_vel=torch.arange(3, 6), proj_gravity=torch.arange(6, 9), joint_pos_body=L(12, ~hm), joint_pos_fingers=L(12, hm), joint_vel_body=L(12 + nj, ~hm), joint_vel_fingers=L(12 + nj, hm),
                     last_action_body=L(12 + 2 * nj, ~hm), last_action_fingers=L(12 + 2 * nj, hm))
        grp_dev = {}
        Hs = torch.stack([OBS[int(hov[i]), i] for i in torch.nonzero(handed & induced).squeeze(-1).tolist()])
        for gname in ("leg", "torso", "arm", "finger"):
            gm = torch.tensor([jgroup(j) == gname for j in jn_]); cols = L(12, gm); dv = (Hs[:, cols] - mu[cols]).abs()
            grp_dev[gname] = dict(n_joints=int(gm.sum()), mean_abs_deviation_rad=round(float(dv.mean()), 3), max_abs_deviation_rad=round(float(dv.max()), 3), control_std_rad_mean=round(float(ref[:, cols].std(0).mean()), 4))
        finger_cols = L(12, hm); raw_f = (torch.stack([OBS[int(hov[i]), i] for i in torch.nonzero(handed & induced).squeeze(-1).tolist()])[:, finger_cols] - mu[finger_cols]).abs()
        finger_raw = dict(mean_abs_deviation_rad_from_control_mean=round(float(raw_f.mean()), 3), max_abs_deviation_rad=round(float(raw_f.max()), 3), control_std_rad_mean=round(float(ref[:, finger_cols].std(0).mean()), 4))
        hidx = torch.nonzero(handed & induced).squeeze(-1); Oh = torch.stack([OBS[int(hov[i]), i] for i in hidx.tolist()]); Z = (Oh - mu) / sd
        def grp(sel_idx):
            if len(sel_idx) == 0: return None
            sub = torch.tensor([j for j, i in enumerate(hidx.tolist()) if i in set(sel_idx)]); z = Z[sub]; az = z.abs().mean(0)
            top = torch.argsort(az, descending=True)
            return dict(n=int(len(sub)), mean_abs_z_by_term={k_: round(float(az[v_].mean()), 2) for k_, v_ in terms.items()},
                        top_dims=[dict(dim=nm[int(j)], mean_abs_z=round(float(az[j]), 2), mean_z=round(float(z[:, j].mean()), 2)) for j in top[:12] if not nm[int(j)].startswith("cmd_")][:10])
        done_i = torch.nonzero(done & induced).squeeze(-1).tolist(); ref_i = torch.nonzero(refell & induced).squeeze(-1).tolist()
        dg = grp(set(done_i)); rg = grp(set(ref_i))
        # dimensions that separate re-fell from completed (Cohen d on the z-scores)
        sep = None
        if dg and rg:
            za = Z[torch.tensor([j for j, i in enumerate(hidx.tolist()) if i in set(done_i)])]; zb = Z[torch.tensor([j for j, i in enumerate(hidx.tolist()) if i in set(ref_i)])]
            pooled = ((za.var(0) * (len(za) - 1) + zb.var(0) * (len(zb) - 1)) / max(len(za) + len(zb) - 2, 1)).sqrt().clamp(min=1e-6); d_ = (zb.mean(0) - za.mean(0)) / pooled
            sep = [dict(dim=nm[int(j)], cohen_d=round(float(d_[j]), 2)) for j in torch.argsort(d_.abs(), descending=True)[:10] if not nm[int(j)].startswith("cmd_")]
        sp_h = torch.stack([TR["stance"].cpu()[int(hov[i]), i] for i in hidx.tolist()]); sp_c = TR["stance"].cpu()[PUSH - 100:PUSH - 1][:, cidx].reshape(-1)
        diag = dict(definition="z = (policy observation at the hand-over step - mean) / std of the no-push control's observations over the 2 s before the push (walking gait); velocity-command dims excluded from rankings",
                    joint_position_deviation_by_group=grp_dev, finger_joint_positions=finger_raw, n_control_samples=int(ref.shape[0]), n_handed_over=int(len(hidx)), all_handed_over=grp(set(hidx.tolist())), completed_walk=dg, refell=rg, refell_vs_completed_cohen_d=sep,
                    foot_spread_m=dict(at_handover_mean=round(float(sp_h.mean()), 3), control_mean=round(float(sp_c.mean()), 3), control_std=round(float(sp_c.std()), 3)))
    ndg = {}
    if ND:
        sel = torch.isfinite(ND["h"].cpu()); Hn = {k_: v_.cpu() for k_, v_ in ND.items()}
        thr = dict(best_run_steps=(">=", HOLD), vel_frac=("<=", 0.02), flight_frac=("<=", 0.02), pos_frac=("<=", 0.10), hard_frac=("<=", 0.03), hard_excess=("<=", 0.10), torque_frac=("<=", 0.03), peak_vz=("<=", 2.0), stance_hold=("<=", 0.60))
        ndg = dict(n=int(sel.sum()), clause_failures={k_: int((((Hn[k_] < v_[1]) if v_[0] == ">=" else (Hn[k_] > v_[1])) & sel).sum()) for k_, v_ in thr.items()},
                   reached_strict_2s_stand=int(((Hn["best_run_steps"] >= HOLD) & sel).sum()), final_pelvis_height_m_mean=float(Hn["h"][sel].mean()), final_up_mean=float(Hn["up"][sel].mean()),
                   best_supported_run_s_mean=float((Hn["best_run_steps"][sel] * DT).mean()), note="evaluated when the recovery window timed out; a clause listed failed for that many timed-out robots")
    def st(x):
        x = np.array([v for v in x if v == v], dtype=float)
        return None if len(x) == 0 else dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(x.max()))
    names = FH.NAMES
    def group(sel):
        n = int(sel.sum())
        if n == 0: return None
        return dict(n=n, push_applied=int((impulsed & sel).sum()), fell_induced=int((induced & sel).sum()), spontaneous_falls=int((spont & sel).sum()), detection_latency_s=st(lat_s[sel & induced].tolist()),
                    stand_gate_pass=int((handed & sel & induced).sum()), handover_rate_of_induced=float((handed & sel & induced).sum() / max((induced & sel).sum(), 1)),
                    getup_time_s=st(getup_s[sel & induced].tolist()), strict_hold_time_s=st(strict_s[sel & induced].tolist()),
                    completed_walk_window=int((done & sel & induced).sum()), refell_after_handover=int((refell & sel & induced).sum()), env_resets=int((resetf & sel).sum()),
                    along_heading_m=st(along[handed & sel & induced].tolist()), walk_time_without_refall_s=st(walk_time[handed & sel & induced].tolist()),
                    official_v3_gate_passed_in_window=int(((v3_first >= 0) & sel & induced).sum()), handover_with_v3_true=int((v3_at & sel & induced).sum()), success_with_v3_true=int((success & v3_at & sel).sum()),
                    walked_10s_and_ge3m=int((ok3 & sel).sum()), time_to_refall_s=st(ttr[sel & induced].tolist()), settle_cancels=int(cancels[sel].sum()),
                    episode_success=int((success & sel).sum()), episode_success_rate_of_induced=float((success & sel).sum() / max((induced & sel).sum(), 1)),
                    outcomes={n_: int(((S == j) & sel).sum()) for j, n_ in enumerate(names)}, chain_failure_reasons={n_: int(((reason == j) & sel & induced).sum()) for j, n_ in enumerate(M.CHAIN_REASONS)})
    by_mag = {str(m_): group(mag_c == m_) for m_ in sorted(set(levels))}
    res = dict(protocol="M2 continuous fall/hand-over (EVAL_PROTOCOL.md Stage 3, DEV, evaluation only)", task=args.task, checkpoint=args.checkpoint, walk_checkpoint=xa.walk_checkpoint, seed=xa.seed, num_envs=N,
               variant=dict(blend_joints=xa.blend_joints, trigger=xa.trigger, ready_hold=xa.ready_hold, settle_steps=xa.settle, settle_mode=xa.settle_mode, blend_steps=xa.blend, ramp_steps=xa.ramp, gate_arg=xa.gate),
               timeline=dict(push_s=xa.push_s, nofall_s=xa.nofall_s, det_steps=xa.det_steps, fall_h=fsm.fall_h, fall_up=fsm.fall_up, recover_max_s=xa.recover_max_s, walk_s=xa.walk_s, fwd_cmd=xa.fwd, total_s=xa.total_s, dt=DT),
               fall_induction=dict(method="horizontal root-velocity impulse added to the base velocity at push_s, random direction per env (seeded)", magnitudes_m_s=levels, magnitude_0="no-push control"),
               gate=("fall_handover.window_gate (criteria v3 stand/hold over the get-up window; 3b self-penetration not applied)" if xa.gate == "v3" else "EXPLORATORY strict stand only (supported 2 s stand, no validity clauses)"), gate_mode=xa.gate, env_resets_total=n_reset_events,
               overall=group(torch.ones(N, dtype=torch.bool)), by_push_magnitude=by_mag,
               no_stand_gate_diagnosis=ndg,
               control_no_push=dict(n=int((mag_c == 0).sum()), fell=int(((mag_c == 0) & entered_rec).sum()), still_walking_at_end=int(((mag_c == 0) & (S == FH.WALK)).sum()),
                                    walk_reference_along_heading_m_over_window=st(c_along.tolist()), reference_ge_3m=int((c_along >= M.P.walk_dist).sum()), reference_ge_5m=int((c_along >= M.P.walk_speed * xa.walk_s).sum()),
                                    walk_from_rest_along_heading_m_over_window=st(r_along.tolist()), from_rest_ge_5m=int((r_along >= M.P.walk_speed * xa.walk_s).sum())),
               handover_state_diagnosis=diag,
               movement_during_getup={k: st(v) for k, v in mv.items()}, movement_note="M1 metrics over [detect, hand-over) per env; spread/hip peak use h>0.12 gating inside the window",
               per_env=dict(mag=mag_c.tolist(), angle_rad=[round(float(a_), 3) for a_ in ang.cpu()], state=[names[int(s_)] for s_ in S], impulse_step=imp.tolist(), fall_onset_step=onset.tolist(), detect_step=det.tolist(), handover_step=hov.tolist(), end_step=endk.tolist(),
                            getup_s=[None if v != v else round(v, 2) for v in getup_s.tolist()], along_m=[round(float(v), 2) for v in along.tolist()], success=success.int().tolist(), trigger_step=trig.tolist(), v3_first_step=v3_first.tolist(), ready_first_step=ready_first.tolist(), v3_true_at_handover=v3_at.int().tolist(), walked10_ge3m=ok3.int().tolist(), time_to_refall_s=[None if v != v else round(v, 2) for v in ttr.tolist()]))
    json.dump(res, open(xa.out, "w"), indent=1); print("M2_JSON", json.dumps({k: res[k] for k in ["overall", "control_no_push", "env_resets_total"]}), flush=True)
    C.close_env(env)
