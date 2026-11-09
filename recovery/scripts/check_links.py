"""Verify body references on known poses. For each pose: world height and up-vector of key links from the articulation data AND from Newton's raw state; rigid-attachment test: the transform of each link
relative to torso_link and to pelvis, across envs and poses. A link rigid on the torso has ~zero spread relative to torso_link; it should vary relative to the pelvis when torso_joint moves.
  python check_links.py --task G1-Recovery-A-v0 --checkpoint c.pt --num_envs 16 [--pose_set s.npz] --seconds 5 --label fold --out o.json physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--pose_set", default=None); ex.add_argument("--seconds", type=float, default=5.0); ex.add_argument("--label", required=True); ex.add_argument("--out", required=True)
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False); env_cfg.fixed_poses_path = os.path.abspath(xa.pose_set) if xa.pose_set else None


def qmul(a, b):
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz], -1)
def qinv(q): return torch.cat([-q[..., :3], q[..., 3:]], -1)
def qrot(q, v):
    u = q[..., :3]; w = q[..., 3:]; t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 30.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False); env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; r = u.scene["robot"]; dev = u.device; N = u.num_envs
    runner = create_rsl_rl_runner(env, agent_cfg)
    if args.checkpoint: runner.load(args.checkpoint, map_location=agent_cfg.device)
    pol = runner.get_inference_policy(device=dev)
    if xa.pose_set:
        f = np.load(xa.pose_set); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    else:
        K.get_levels(u)[:] = 0; u._rec_level_fixed = True
    obs, _ = env.reset()
    for s in range(int(xa.seconds / u.step_dt)):
        with torch.inference_mode(): obs, _, _, _ = env.step(pol(obs))
    names = list(r.body_names); want = [n for n in names if n in ("pelvis", "torso_link", "head_link", "imu_link", "d435_link", "mid360_link", "logo_link") or "shoulder_pitch_link" in n or "shoulder_roll_link" in n or "ankle_roll_link" in n or n in ("left_hip_pitch_link", "left_knee_link")]
    d = r.data; idx = {n: names.index(n) for n in want}
    pos = (d.body_link_pos_w.torch if hasattr(d, "body_link_pos_w") else d.body_pos_w.torch); quat = (d.body_link_quat_w.torch if hasattr(d, "body_link_quat_w") else d.body_quat_w.torch)
    pos_alt = d.body_pos_w.torch; com = d.body_com_pos_w.torch if hasattr(d, "body_com_pos_w") else None
    oz = u.scene.env_origins[:, 2:3]
    ez = torch.tensor([0., 0., 1.], device=dev).expand(N, 3)
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    m = NM.get_model(); st = NM.backend.state_0 if hasattr(NM, "backend") else NM.get_state_0()
    bq = wp.to_torch(st.body_q); labels = list(getattr(m, "body_label", None) or getattr(m, "body_key", None) or [])
    res = dict(label=xa.label, checkpoint=args.checkpoint, seconds=xa.seconds, num_envs=N, all_body_names=names, torso_joint_mean=float(d.joint_pos.torch[:, r.joint_names.index("torso_joint")].mean()), torso_joint_std=float(d.joint_pos.torch[:, r.joint_names.index("torso_joint")].std()), links={})
    pi, ti = idx["pelvis"], idx["torso_link"]
    for n, i in idx.items():
        p = pos[:, i]; q = quat[:, i]; up = qrot(q, ez)[:, 2]
        rel_t = qrot(qinv(quat[:, ti]), p - pos[:, ti]); rel_p = qrot(qinv(quat[:, pi]), p - pos[:, pi])
        rq_t = qmul(qinv(quat[:, ti]), q); ang_t = 2 * torch.atan2(rq_t[:, :3].norm(dim=-1), rq_t[:, 3].abs())
        rq_p = qmul(qinv(quat[:, pi]), q); ang_p = 2 * torch.atan2(rq_p[:, :3].norm(dim=-1), rq_p[:, 3].abs())
        e = dict(z_mean=float((p[:, 2:3] - oz).mean()), z_alt_body_pos_w=float((pos_alt[:, i, 2:3] - oz).mean()), up_z_mean=float(up.mean()), up_z_std=float(up.std()),
                 rel_to_torso_xyz=[round(float(x), 4) for x in rel_t.mean(0)], rel_to_torso_std=round(float(rel_t.std(0).max()), 5), rel_to_torso_angle_mean=round(float(ang_t.mean()), 4), rel_to_torso_angle_std=round(float(ang_t.std()), 5),
                 rel_to_pelvis_xyz=[round(float(x), 4) for x in rel_p.mean(0)], rel_to_pelvis_std=round(float(rel_p.std(0).max()), 5), rel_to_pelvis_angle_mean=round(float(ang_p.mean()), 4), rel_to_pelvis_angle_std=round(float(ang_p.std()), 5))
        if com is not None: e["com_z_mean"] = float((com[:, i, 2:3] - oz).mean())
        k = [j for j, l in enumerate(labels) if l.endswith("/" + n) and "env_0/" in l]
        if k: e["newton_raw_z_env0"] = round(float(bq[k[0], 2]), 4); e["articulation_z_env0"] = round(float(p[0, 2]), 4)
        res["links"][n] = e
    json.dump(res, open(xa.out, "w"), indent=1); print("LINKS_OK", flush=True); C.close_env(env)
