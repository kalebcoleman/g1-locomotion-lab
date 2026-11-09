"""At the first moment each held-out env meets the standing criterion, record whole-body state, then see whether it keeps standing for 2 s.
Compares envs that hold (success) with those that stand then fall. Features: pelvis vertical/horizontal speed, torso angular speed, COM horizontal
offset from the feet's support centre, foot separation, fraction of feet in contact, hand/arm contact force, joint-velocity norm, time of first stand.
Unassisted. python diag_standfail.py --task G1-Recovery-A-v0 --checkpoint m.pt --num_envs 256 --out d.json physics=newton_mjwarp"""
import sys, os, json, argparse
import numpy as np
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out", required=True)
ex.add_argument("--heldout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "heldout_poses.npz"))
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from g1_recovery import common as K
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False); env_cfg.fixed_poses_path = os.path.abspath(xa.heldout)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 20.0
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; dev = u.device; N = u.num_envs; r = u.scene["robot"]
    runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device); policy = runner.get_inference_policy(device=dev)
    f = np.load(xa.heldout); u._rec_fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    obs, _ = env.reset(); t = K.get_tracker(u)
    sens = u.scene.sensors.get("contact_forces"); bn = list(sens.body_names)
    feet_i = [i for i, n in enumerate(bn) if "ankle_roll" in n]; arm_i = [i for i, n in enumerate(bn) if any(k in n for k in ("wrist", "elbow", "shoulder"))]
    rb = list(r.body_names); rfeet = [rb.index(n) for n in rb if "ankle_roll" in n]
    mass = r.data.body_mass.torch[0].to(dev); M = mass.sum()
    snap = {}; first = torch.full((N,), -1, device=dev, dtype=torch.long)
    steps = int(K.EPISODE_S / u.step_dt)
    for s in range(steps):
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        newly = t.stand_now & (first < 0)
        if newly.any():
            idx = torch.nonzero(newly).squeeze(-1)
            com = (r.data.body_com_pos_w.torch[idx] * mass[None, :, None]).sum(1) / M if hasattr(r.data, "body_com_pos_w") else r.data.root_pos_w.torch[idx]
            fp = r.data.body_pos_w.torch[idx][:, rfeet]; centre = fp.mean(1)
            fv = torch.linalg.norm(r.data.body_lin_vel_w.torch[idx][:, :, :2] if hasattr(r.data, "body_lin_vel_w") else r.data.root_lin_vel_w.torch[idx][:, None, :2], dim=-1)
            ff = torch.linalg.norm(sens.data.net_forces_w.torch[idx], dim=-1)
            for j, e in enumerate(idx.tolist()):
                snap[e] = dict(step=s, com_off_xy=float(torch.linalg.norm((com[j] - centre[j])[:2])), com_z=float(com[j, 2] - u.scene.env_origins[e, 2]),
                               foot_sep=float(torch.linalg.norm((fp[j, 0] - fp[j, 1])[:2])), foot_z=float(fp[j, :, 2].mean() - u.scene.env_origins[e, 2]),
                               pelvis_vz=float(r.data.root_lin_vel_w.torch[e, 2]), pelvis_vxy=float(torch.linalg.norm(r.data.root_lin_vel_w.torch[e, :2])),
                               torso_angvel=float(torch.linalg.norm(r.data.root_ang_vel_w.torch[e])), feet_contact=float((ff[j, feet_i] > 10).float().mean()),
                               arm_force=float(ff[j, arm_i].sum()), jvel_norm=float(torch.linalg.norm(r.data.joint_vel.torch[e])))
            first[idx] = s
    fc = t.failure_cause().cpu(); succ = t.succ.cpu()
    def agg(sel):
        v = [snap[e] for e in sel if e in snap]
        if not v: return None
        return {k: round(float(np.mean([x[k] for x in v])), 3) for k in v[0]} | {"n": len(v)}
    ok = [e for e in range(N) if succ[e]]; fell = [e for e in range(N) if int(fc[e]) == 3]
    res = dict(task=args.task, checkpoint=args.checkpoint, held_out_envs=N, n_success=len(ok), n_stood_then_fell=len(fell), at_first_stand={"success": agg(ok), "stood_then_fell": agg(fell)},
               note="features sampled at the first step the standing criterion is met (pelvis>=0.60 m, tilt<=26 deg); com_off_xy = horizontal distance between whole-body COM and the midpoint of the two feet (m)")
    json.dump(res, open(xa.out, "w"), indent=1); print("STANDFAIL_JSON", json.dumps(res), flush=True)
    C.close_env(env)
