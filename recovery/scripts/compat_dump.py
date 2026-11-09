"""Dump the facts that decide whether the walking policy and a recovery policy are compatible, for ONE task. Run once per task, then diff the JSONs.
  python compat_dump.py --task Isaac-Velocity-Flat-G1|G1-Recovery-A-v0 --out x.json --num_envs 2 physics=newton_mjwarp"""
import sys, os, json, argparse
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
ex = argparse.ArgumentParser(add_help=False); ex.add_argument("--out", required=True)
xa, rest = ex.parse_known_args(sys.argv[1:]); args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa
from isaaclab_tasks.utils import resolve_task_config
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    u = env.unwrapped; r = u.scene["robot"]; d = r.data; env.reset()
    om = u.observation_manager; am = u.action_manager
    t = am.get_term("joint_pos")
    def arr(x):
        if isinstance(x, (int, float)): return [float(x)]
        x = x.torch if hasattr(x, 'torch') else x
        return [round(float(v), 5) for v in torch.as_tensor(x).reshape(-1).tolist()]
    obs = om.compute()
    out = dict(task=args.task, joint_names=list(r.joint_names), body_names=list(r.body_names), num_envs=u.num_envs,
        obs_terms={g: list(om.active_terms[g]) for g in om.active_terms}, obs_term_dims={g: [list(x) for x in om.group_obs_term_dim[g]] for g in om.group_obs_term_dim},
        obs_noise_enabled=bool(om.cfg.policy.enable_corruption) if hasattr(om.cfg, "policy") else None, policy_obs_dim=int(obs["policy"].shape[-1]),
        action_dim=int(am.total_action_dim), action_scale=arr(t._scale) if hasattr(t, "_scale") else None, action_offset=arr(t._offset) if hasattr(t, "_offset") else None,
        action_clip=str(getattr(t.cfg, "clip", None)), use_default_offset=getattr(t.cfg, "use_default_offset", None),
        default_joint_pos=arr(d.default_joint_pos.torch[0]), soft_pos_limits=arr(d.soft_joint_pos_limits.torch[0]), soft_vel_limits=arr(d.soft_joint_vel_limits.torch[0]),
        joint_stiffness=arr(d.joint_stiffness.torch[0]) if hasattr(d, "joint_stiffness") else None, joint_damping=arr(d.joint_damping.torch[0]) if hasattr(d, "joint_damping") else None,
        effort_limits=arr(d.joint_effort_limits.torch[0]) if hasattr(d, "joint_effort_limits") else None,
        sim_dt=u.cfg.sim.dt, decimation=u.cfg.decimation, step_dt=u.step_dt, episode_length_s=u.cfg.episode_length_s, robot_usd=str(env_cfg.scene.robot.spawn.usd_path),
        total_mass=float(d.body_mass.torch[0].sum()), command_ranges={k: str(v) for k, v in vars(env_cfg.commands.base_velocity.ranges).items()},
        rel_standing_envs=env_cfg.commands.base_velocity.rel_standing_envs,
        contact_sensor_bodies=list(u.scene.sensors["contact_forces"].body_names) if "contact_forces" in u.scene.sensors else None)
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    out["num_collision_shapes_per_robot"] = int((NM.get_model().shape_count - 1) // u.num_envs)
    out["policy_obs_first_row"] = arr(obs["policy"][0])
    json.dump(out, open(xa.out, "w"), indent=1); print("COMPAT_OK", xa.out, flush=True)
    C.close_env(env)
