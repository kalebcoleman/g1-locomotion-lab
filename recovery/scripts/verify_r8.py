"""Check the new R8 terms on real simulator states before a long run.
  run.sh verify_r8.py --task G1-Recovery-R8s-v0|R8f-v0|R8p-v0 --checkpoint <model.pt> --num_envs 128 physics=newton_mjwarp"""
import sys, json
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C
args = P._parse_args(sys.argv[1:]); args.train_env_cfg = True
import g1_recovery  # noqa: F401
from g1_recovery import common as K, lane_a as A
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version
env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args); agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg); env_cfg.seed = 1; env_cfg.episode_length_s = 20.0
    env = RslRlVecEnvWrapper(C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False), clip_actions=agent_cfg.clip_actions)
    u = env.unwrapped; r = u.scene["robot"]; runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint, map_location=agent_cfg.device); policy = runner.get_inference_policy(device=u.device)
    obs, _ = env.reset(); term = u.action_manager.get_term("joint_pos"); rows = []; targ_err = []; tau_settle = []; tau_after = []
    up0 = -r.data.projected_gravity_b.torch[:, 2]
    # body-frame gravity direction right after reset tells the family: prone means the robot's front faces down (+x body axis points down)
    g0 = r.data.projected_gravity_b.torch.clone()
    for s in range(300):
        with torch.inference_mode(): obs, _, _, _ = env.step(policy(obs))
        tgt = term.processed_actions; q = r.data.joint_pos.torch[:, term._joint_ids]; tq = r.data.applied_torque.torch.abs().max(-1).values.mean().item()
        (tau_settle if s < 30 else tau_after).append(tq); 
        if s < 30: targ_err.append((tgt - q).abs().max().item())
        if s % 50 == 49:
            sup = K.standing_mask(u) & K.supported_mask(u); fs = A.feet_load_share(u); pr = A.prop_force(u); bonus = A.standing_bonus_supported(u); leg = K.standing_mask(u)
            rows.append(dict(step=s + 1, stand_mask=int(leg.sum()), supported=int(sup.sum()), bonus_max_when_unsupported=float(bonus[~sup].max()) if (~sup).any() else 0.0, feet_share_mean=round(float(fs.mean()), 3), feet_share_when_supported=round(float(fs[sup].mean()), 3) if sup.any() else None, feet_share_max=round(float(fs.max()), 3), prop_mean=round(float(pr.mean()), 3), prop_nonzero=int((pr > 0).sum()), prop_max=round(float(pr.max()), 3), stance_pen_mean=round(float(A.stance_excess(u).mean()), 3), stance_pen_supported=round(float(A.stance_excess(u)[sup].mean()), 3) if sup.any() else None))
    print("VERIFY_JSON", json.dumps(dict(task=args.task, settle_steps=getattr(term.cfg, "settle_steps", 0), first30_target_minus_q_at_last_substep_max=round(max(targ_err), 4) if targ_err else None, tau_mean_of_env_max_first30=round(sum(tau_settle) / len(tau_settle), 2), tau_mean_of_env_max_after=round(sum(tau_after) / len(tau_after), 2),
          gravity_body_x_mean_at_reset=round(float(g0[:, 0].mean()), 3), gravity_body_y_abs_mean=round(float(g0[:, 1].abs().mean()), 3), rows=rows)), flush=True)
    C.close_env(env)
