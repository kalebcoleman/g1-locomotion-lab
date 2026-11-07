"""Train a G1 recovery policy (Lane A or B) with rsl_rl PPO. Run from the Isaac Lab checkout with PYTHONPATH=<g1-recovery>.
  python train_recovery.py --task G1-Recovery-A-v0 --num_envs 1024 --max_iterations 1500 --seed 1 --out_root <dir> --run_name a_s1 physics=newton_mjwarp
Writes <out_root>/<run_name>/ (checkpoints, tensorboard, configs, train_summary.json)."""
import sys, os, json, time, argparse
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--max_iterations", type=int, default=1500)
ex.add_argument("--out_root", required=True)
ex.add_argument("--run_name", required=True)
ex.add_argument("--save_interval", type=int, default=50)
ex.add_argument("--init_actor", default=None, help="warm-start ONLY the actor network from this checkpoint (e.g. the walking policy); critic/optimizer/iteration stay fresh")
ex.add_argument("--lr", type=float, default=None, help="override PPO learning rate (disables the adaptive schedule when given)")
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest)
args.train_env_cfg = True
import g1_recovery  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=False)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    seed = args.seed if args.seed is not None else 1
    agent_cfg.seed = seed; env_cfg.seed = seed
    if xa.lr is not None:
        agent_cfg.algorithm.learning_rate = xa.lr; agent_cfg.algorithm.schedule = "fixed"
    agent_cfg.max_iterations = xa.max_iterations; agent_cfg.save_interval = xa.save_interval
    log_dir = os.path.abspath(os.path.join(xa.out_root, xa.run_name)); os.makedirs(log_dir, exist_ok=True)
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
    if xa.init_actor:
        runner.load(xa.init_actor, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False}, map_location=agent_cfg.device)
        print("INIT_ACTOR_FROM", xa.init_actor, flush=True)
    u = env.unwrapped
    t0 = time.time()
    runner.learn(num_learning_iterations=xa.max_iterations, init_at_random_ep_len=True)
    wall = time.time() - t0
    n_trans = xa.max_iterations * u.num_envs * agent_cfg.num_steps_per_env
    lv = getattr(u, "_rec_level", None)
    json.dump(dict(init_actor=xa.init_actor, lr_override=xa.lr, task=args.task, num_envs=u.num_envs, iterations=xa.max_iterations, seed=seed, transitions=n_trans, train_wall_s=wall,
                   transitions_per_s=n_trans / wall, mean_level_end=None if lv is None else float(lv.float().mean()),
                   note="wall time/throughput measured on a GPU shared with other jobs; not a benchmark"),
              open(os.path.join(log_dir, "train_summary.json"), "w"), indent=1)
    print("TRAIN_DONE", log_dir, flush=True)
    C.close_env(env)
