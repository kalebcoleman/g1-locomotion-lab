"""Short diagnostic: does Isaac-Velocity-Flat-G1 really run Newton/MuJoCo-Warp physics on the GPU?
Run from the Isaac Lab checkout (see RUN.md). 16 envs, ~60 env steps, headless, no cameras. Writes evidence/*.
"""
import sys, os, json, time, argparse, collections, platform
import warp as wp
wp.config.enable_backward = False
import torch
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--evidence", required=True)
ex.add_argument("--warmup", type=int, default=30)
ex.add_argument("--traced", type=int, default=20)
xa, rest = ex.parse_known_args(sys.argv[1:])
os.makedirs(xa.evidence, exist_ok=True)
E = lambda n: os.path.join(xa.evidence, n)
args = P._parse_args(rest)
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg, check_rsl_rl_version

rep = {}                      # everything we learn goes in here -> evidence/report.json
def dev(o):
    d = getattr(o, "device", None); return None if d is None else str(d)
def info_arr(a):
    if a is None: return None
    try:
        return dict(type=type(a).__name__, device=dev(a), shape=list(a.shape), dtype=str(a.dtype), ptr=int(a.ptr) if hasattr(a, "ptr") else int(a.data_ptr()))
    except Exception as e: return dict(type=type(a).__name__, err=repr(e))

env_cfg, agent_cfg = resolve_task_config(args.task, args.agent, play_mode=not args.train_env_cfg)
with launch_simulation(env_cfg, args):
    agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = 1
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = create_rsl_rl_runner(env, agent_cfg); runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    u = env.unwrapped
    from isaaclab_newton.physics.newton_manager import NewtonManager as NM
    from isaaclab.physics import PhysicsManager as PM

    # ---------- 1. what was configured / resolved ----------
    pc = u.cfg.sim.physics
    rep["configured"] = dict(
        python=platform.python_version(), warp=wp.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
        sim_device_cfg=str(u.cfg.sim.device), env_device=str(u.device), num_envs=u.num_envs,
        physics_cfg_class=type(pc).__name__, physics_class_type=str(getattr(pc, "class_type", None)),
        solver_cfg=repr(getattr(pc, "solver_cfg", None)), use_cuda_graph=getattr(pc, "use_cuda_graph", None),
        physics_manager_class=PM.__name__, PhysicsManager__device=str(PM._device), active_manager=type(NM).__name__,
        sim_dt=u.cfg.sim.dt, decimation=u.cfg.decimation, step_dt=u.step_dt,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        warp_devices=[str(d) for d in wp.get_devices()], warp_cuda_device_count=wp.get_cuda_device_count(),
        warp_default_device=str(wp.get_device()),
        torch_device_name=torch.cuda.get_device_name(0), torch_cuda_available=torch.cuda.is_available())
    try:
        import newton, mujoco_warp, mujoco
        rep["configured"].update(newton=newton.__version__, mujoco_warp=getattr(mujoco_warp, "__version__", "n/a"), mujoco=mujoco.__version__)
    except Exception as e: rep["configured"]["versions_err"] = repr(e)

    # ---------- 2. instantiated solver and where the physics arrays live ----------
    solver = NM._solver; model = NM.get_model(); st0 = NM.backend.state_0 if hasattr(NM, "backend") else NM.get_state_0()
    s = dict(solver_class=f"{type(solver).__module__}.{type(solver).__qualname__}", solver_device=dev(solver),
             model_device=dev(model), graph_present=NM._graph is not None, graph_type=type(NM._graph).__name__,
             all_graphable=NM._is_all_graphable(), num_substeps=NM._num_substeps, decimation=NM._decimation,
             needs_collision_pipeline=NM._needs_collision_pipeline)
    for nm in ["body_q", "body_qd", "joint_q", "joint_qd", "joint_target_pos", "body_mass", "joint_X_p"]:
        s["model."+nm] = info_arr(getattr(model, nm, None))
    for nm in ["body_q", "body_qd", "joint_q", "joint_qd"]:
        s["state_0."+nm] = info_arr(getattr(st0, nm, None))
    ctl = NM.backend.control if hasattr(NM, "backend") else None
    if ctl is not None: s["control.joint_f"] = info_arr(getattr(ctl, "joint_f", None))
    mjw_model = getattr(solver, "mjw_model", None); mjw_data = getattr(solver, "mjw_data", None)
    s["mjw_model_type"] = f"{type(mjw_model).__module__}.{type(mjw_model).__qualname__}"
    s["mjw_data_type"] = f"{type(mjw_data).__module__}.{type(mjw_data).__qualname__}"
    for nm in ["qpos", "qvel", "qacc", "xpos", "contact", "efc", "nworld", "naconmax", "njmax"]:
        v = getattr(mjw_data, nm, None)
        if hasattr(v, "dist") or hasattr(v, "shape") and hasattr(v, "device"):
            s["mjw_data."+nm] = info_arr(v)
        elif v is not None:
            s["mjw_data."+nm] = repr(v)[:200]
    for nm in ["nworld", "nv", "nq", "nbody", "ngeom", "dof_damping", "body_mass"]:
        v = getattr(mjw_model, nm, None)
        s["mjw_model."+nm] = info_arr(v) if hasattr(v, "device") else repr(v)[:100]
    # solver-owned warp arrays (any attribute that is a wp.array)
    s["solver_wp_arrays_devices"] = dict(collections.Counter(
        dev(v) for v in vars(solver).values() if isinstance(v, wp.array)))
    # robot data seen by the RL layer: Warp array vs its torch view (same memory?)
    r = u.scene["robot"].data
    proxy = r.root_lin_vel_b; wpa = proxy.warp; tv = proxy.torch; s['robot.root_lin_vel_b.proxy_type'] = type(proxy).__name__
    s["robot.root_lin_vel_b"] = info_arr(wpa); s["robot.root_lin_vel_b.torch"] = dict(device=str(tv.device), data_ptr=int(tv.data_ptr()), same_memory_as_warp=int(tv.data_ptr()) == int(wpa.ptr))
    rep["physics_state"] = s

    obs = env.get_observations()
    with torch.inference_mode(): act = policy(obs)
    rep["policy_tensors"] = dict(obs_policy_device=str(obs["policy"].device), obs_shape=list(obs["policy"].shape),
        action_device=str(act.device), policy_params_device=str(next(runner.alg.actor.parameters()).device),
        note="PyTorch tensors of the RL layer/policy; separate from Warp physics arrays unless a pointer matches")

    # ---------- 3. warm-up, graph content, CUDA trace ----------
    def step():
        global obs
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
    for _ in range(xa.warmup): step()
    wp.synchronize(); torch.cuda.synchronize()
    rep["graph_after_warmup"] = dict(present=NM._graph is not None, use_graph_flag=bool(getattr(pc, "use_cuda_graph", False)))
    try:
        wp.capture_debug_dot_print(NM._graph, E("newton_graph.dot"), verbose=True)
        rep["graph_after_warmup"]["dot_file"] = "newton_graph.dot"
    except Exception as e: rep["graph_after_warmup"]["dot_err"] = repr(e)

    from torch.profiler import profile, ProfilerActivity
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(xa.traced): step()
        wp.synchronize(); torch.cuda.synchronize()
    rep["trace"] = dict(env_steps_traced=xa.traced, wall_s=time.time() - t0, note="shared GPU; timing is not a benchmark")
    try: prof.export_chrome_trace(E("trace_chrome.json"))
    except Exception as e: rep["trace"]["chrome_err"] = repr(e)
    kern = collections.defaultdict(lambda: [0, 0.0]); rt = collections.Counter(); other = collections.Counter()
    for ev in prof.events():
        dt = str(ev.device_type)
        if "CUDA" in dt:
            kern[ev.name][0] += 1; kern[ev.name][1] += ev.device_time if hasattr(ev, "device_time") else ev.cuda_time
        if ev.name in ("cudaGraphLaunch", "cuGraphLaunch", "cudaLaunchKernel", "cuLaunchKernel", "cudaMemcpyAsync", "cudaStreamSynchronize"): rt[ev.name] += 1
    json.dump({k: dict(count=v[0], total_us=v[1]) for k, v in kern.items()}, open(E("traced_kernels.json"), "w"), indent=1)
    rep["trace"]["distinct_gpu_kernels"] = len(kern); rep["trace"]["gpu_kernel_launches"] = sum(v[0] for v in kern.values())
    rep["trace"]["gpu_time_us_total"] = sum(v[1] for v in kern.values()); rep["trace"]["runtime_api_counts"] = dict(rt)
    open(E("key_averages_cuda.txt"), "w").write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=90))
    json.dump(rep, open(E("report.json"), "w"), indent=1, default=str)
    print("REPORT_WRITTEN", E("report.json"), flush=True)
    C.close_env(env)
