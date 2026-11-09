"""Per-step trace of a recovery checkpoint on a fixed pose set, used to audit any apparent success. Reads state only; applies nothing.
  run.sh diag_trace.py --task G1-Recovery-A-Assist-v0 --checkpoint <model.pt> --num_envs 256 --heldout <set.npz> --out <file.npz> physics=newton_mjwarp   (G1R_EVAL_ASSIST=0)
Saves (T, N) traces: pelvis height, up, 3D speed, vz, contact force by body group, foot height, stance width, joint limit excess, torque ratio against Unitree's published limits,
body poses every 5th step for an offline self-penetration check, and the same v2 / v2.1 tracker verdicts as eval_recovery.py, plus the collision geometry of one robot."""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out", required=True); ex.add_argument("--heldout", required=True)
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K, metrics as M
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
env_cfg.fixed_poses_path = os.path.abspath(xa.heldout)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1; env_cfg.episode_length_s = 20.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]
    from g1_recovery import host_port as _HP
    if _HP.is_host_checkpoint(args.checkpoint):
        policy = _HP.load_host_policy(args.checkpoint, dev)      # HoST-port actor (train_host.py)
    else:
        runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device)
        policy = runner.get_inference_policy(device=dev)
    f = np.load(xa.heldout); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    obs, _ = env.reset()
    steps = int(round(K.EPISODE_S / u.step_dt)); t = K.get_tracker(u)
    sens = u.scene.sensors["contact_forces"]; bn = list(sens.body_names)
    grp = lambda *keys: [i for i, n in enumerate(bn) if any(k in n for k in keys)]
    G = dict(foot_l=[bn.index("left_ankle_roll_link")], foot_r=[bn.index("right_ankle_roll_link")],
             hand=grp("palm", "zero", "one", "two", "three", "four", "five", "six"), arm=grp("elbow", "shoulder"), knee=grp("knee"), trunk=grp("pelvis", "torso", "head", "imu", "logo"), hip_ankle=grp("hip", "ankle_pitch"))
    fb = [r.body_names.index("left_ankle_roll_link"), r.body_names.index("right_ankle_roll_link")]
    lim_real = torch.full((len(r.joint_names),), float("inf"), device=dev)
    for i, n in enumerate(r.joint_names):
        if not K.is_hand_joint(n):
            lim_real[i] = M.real_torque_limit(n)
    body_j = torch.tensor([not K.is_hand_joint(n) for n in r.joint_names], device=dev)
    org = u.scene.env_origins
    tr = {k: [] for k in ["h", "up", "spd", "vz", "foot_z_max", "stance", "soft_ex", "hard_ex", "tau_ratio", "supp", "x", "y", "tau_over"] + ["f_" + g for g in G]}
    body_p, body_q = [], []; tau_max = torch.zeros(N, len(r.joint_names), device=dev); tau_over_j = torch.zeros(N, len(r.joint_names), device=dev); hard_j = torch.zeros(N, len(r.joint_names), device=dev); hard_ex_j = torch.zeros(N, len(r.joint_names), device=dev); q_end = None
    _hip = [r.joint_names.index(n) for n in ("left_hip_roll_joint", "right_hip_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint")]; _bji = torch.nonzero(body_j.reshape(-1, body_j.shape[-1])[0] if body_j.dim() > 1 else body_j).squeeze(-1); _pa = None; _pv = None   # M1 movement traces
    for k in ["hip_roll_l", "hip_roll_r", "hip_yaw_l", "hip_yaw_r", "act_rate", "acc_rms"]: tr[k] = []
    for s in range(steps):
        with torch.inference_mode():
            _act = policy(obs)
            obs, _, _, _ = env.step(_act)
        h = r.data.root_pos_w.torch[:, 2] - org[:, 2]; up = -r.data.projected_gravity_b.torch[:, 2]; v = r.data.root_lin_vel_w.torch
        F = torch.linalg.norm(sens.data.net_forces_w.torch, dim=-1); q = r.data.joint_pos.torch
        pf = r.data.body_pos_w.torch[:, fb]; sl = r.data.soft_joint_pos_limits.torch; hl = r.data.joint_pos_limits.torch
        tau = r.data.applied_torque.torch.abs(); ratio = tau / lim_real
        over = (ratio > 1.0) & body_j
        tau_max = torch.maximum(tau_max, tau); tau_over_j += over.float()
        _hx = torch.maximum(hl[..., 0] - q, q - hl[..., 1]).clamp(min=0) * body_j; hard_j += (_hx > 0).float(); hard_ex_j = torch.maximum(hard_ex_j, _hx); q_end = q.clone()
        _q4 = q[:, _hip]; [tr[k].append(_q4[:, i]) for i, k in enumerate(["hip_roll_l", "hip_roll_r", "hip_yaw_l", "hip_yaw_r"])]
        _jv = r.data.joint_vel.torch[:, _bji]
        tr["act_rate"].append(torch.zeros(N, device=dev) if _pa is None else (_act - _pa).abs().mean(-1)); tr["acc_rms"].append(torch.zeros(N, device=dev) if _pv is None else (((_jv - _pv) / u.step_dt) ** 2).mean(-1).sqrt()); _pa = _act.clone(); _pv = _jv.clone()
        tr["h"].append(h); tr["up"].append(up); tr["spd"].append(torch.linalg.norm(v, dim=-1)); tr["vz"].append(v[:, 2])
        tr["foot_z_max"].append((pf[:, :, 2] - org[:, 2:3]).max(-1).values); tr["stance"].append(torch.linalg.norm(pf[:, 0, :2] - pf[:, 1, :2], dim=-1))
        tr["soft_ex"].append((torch.maximum(sl[..., 0] - q, q - sl[..., 1]).clamp(min=0) * body_j).max(-1).values)
        tr["hard_ex"].append((torch.maximum(hl[..., 0] - q, q - hl[..., 1]).clamp(min=0) * body_j).max(-1).values)
        tr["tau_ratio"].append((ratio * body_j).max(-1).values); tr["tau_over"].append(over.any(-1).float())
        tr["supp"].append(K.supported_mask(u) & (h >= K.SUCCESS_H) & (up >= K.SUCCESS_TILT_COS))
        tr["x"].append(r.data.root_pos_w.torch[:, 0] - org[:, 0]); tr["y"].append(r.data.root_pos_w.torch[:, 1] - org[:, 1])
        for g, idx in G.items():
            tr["f_" + g].append(F[:, idx].sum(-1))
        if s % 5 == 0:
            body_p.append((r.data.body_pos_w.torch - org[:, None]).float().cpu().numpy()); body_q.append(r.data.body_quat_w.torch.float().cpu().numpy())
    n_steps = steps; af = u._assist_log["force_steps"] > 0 if hasattr(u, "_assist_log") else None
    v2 = M.plausible_recovery(t.succ_s, t.vel_viol_steps, t.flight_steps, t.viol_steps, t.peak_vz, n_steps, af)
    v21 = v2 & M.hard_limits_ok(t.hard_viol_steps, t.hard_excess_max, n_steps)
    out = {k: torch.stack(v, 0).float().cpu().numpy().astype(np.float32) for k, v in tr.items()}
    out.update(strict=t.succ_s.cpu().numpy(), v2=v2.cpu().numpy(), v21=v21.cpu().numpy(), succ_start_s=t.succ_start_s.cpu().numpy(), assist_force_steps=(u._assist_log["force_steps"].cpu().numpy() if af is not None else np.zeros(N)),
               vel_frac=(t.vel_viol_steps.float() / n_steps).cpu().numpy(), flight_frac=(t.flight_steps.float() / n_steps).cpu().numpy(), soft_frac=(t.viol_steps.float() / n_steps).cpu().numpy(),
               hard_frac=(t.hard_viol_steps.float() / n_steps).cpu().numpy(), hard_excess=t.hard_excess_max.cpu().numpy(), peak_vz=t.peak_vz.cpu().numpy(), tau_max=tau_max.cpu().numpy(), tau_over_frac_j=(tau_over_j / n_steps).cpu().numpy(), hard_frac_j=(hard_j / n_steps).cpu().numpy(), hard_ex_j=hard_ex_j.cpu().numpy(), q_end=q_end.cpu().numpy(), hard_lo=r.data.joint_pos_limits.torch[0, :, 0].cpu().numpy(), hard_hi=r.data.joint_pos_limits.torch[0, :, 1].cpu().numpy(),
               body_p=np.stack(body_p).astype(np.float32), body_q=np.stack(body_q).astype(np.float32), category=u._rec_fixed["category"][torch.arange(N, device=dev) % u._rec_fixed["z"].shape[0]].cpu().numpy(), dt=u.step_dt,
               joint_names=np.array(r.joint_names), body_names=np.array(r.body_names), lim_real=lim_real.cpu().numpy())
    # collision geometry of one robot, for the offline self-penetration check
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    m = NM.get_model(); sb = m.shape_body.numpy(); st = m.shape_type.numpy(); ss = m.shape_scale.numpy(); sx = m.shape_transform.numpy(); bl = list(m.body_label); sl_ = list(m.shape_label)
    first = [i for i in range(len(sb)) if sb[i] >= 0 and ("env_0/" in bl[sb[i]] or "env_0" in bl[sb[i]].split("/"))]
    out.update(geom_type=st[first], geom_scale=ss[first], geom_xform=sx[first], geom_body=np.array([bl[sb[i]].split("/")[-1] for i in first]), geom_label=np.array([sl_[i] for i in first]))
    src = list(m.shape_source); vs, off = [], [0]; fs, foff = [], [0]
    for i in first:
        sr = src[i] if i < len(src) else None
        v = np.asarray(sr.vertices, dtype=np.float32) if sr is not None and hasattr(sr, "vertices") else np.zeros((0, 3), np.float32); fi = np.asarray(sr.indices, dtype=np.int32) if sr is not None and hasattr(sr, "indices") else np.zeros((0,), np.int32)
        vs.append(v); off.append(off[-1] + len(v)); fs.append(fi); foff.append(foff[-1] + len(fi))
    out.update(geom_verts=np.concatenate(vs), geom_vert_off=np.array(off), geom_faces=np.concatenate(fs), geom_face_off=np.array(foff))
    np.savez_compressed(xa.out, **out)
    print("TRACE_DONE", xa.out, "strict", int(t.succ_s.sum()), "v2", int(v2.sum()), "v21", int(v21.sum()), "geoms", len(first), flush=True)
    C.close_env(env)
