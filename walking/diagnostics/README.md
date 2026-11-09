# GPU-physics diagnostic (separate from the training run)

A short check that the `Isaac-Velocity-Flat-G1` task really steps Newton/MuJoCo-Warp physics on the GPU.
**It is a separate process, not the original training run:** 16 environments, 30 warm-up steps, 20 traced environment
steps, headless, on a GPU shared with other jobs, loading the iteration-1499 checkpoint for the policy. The original
1,024-environment training run was not traced. Timings in these files are not a benchmark and support no speed comparison.

What it found (files in this folder):

- **Configured** (`report.json`): `NewtonCfg` / `NewtonMJWarpManager`, solver `mujoco_warp`, `use_mujoco_cpu=False`,
  `use_cuda_graph=True`, device `cuda:0` (one GPU).
- **State on CUDA** (`report.json`): the instantiated `SolverMuJoCo`, the Newton model/state arrays and the
  mujoco_warp `Model`/`Data` arrays are on `cuda:0`. The policy's PyTorch tensors are separate; the one bridge checked
  (a robot-data view) shares memory with its Warp array.
- **Traced** (`traced_kernels.json`, `graph_vs_trace.json`): a torch.profiler/CUPTI trace after warm-up showed 80 CUDA graph
  launches over 20 env steps (4 physics steps per env step). The captured graph was dumped (`newton_graph.dot`: 244 kernel
  nodes, 85 distinct kernels incl. Newton collision and MuJoCo-Warp solver kernels), and all 85 appear in the trace. Actuator, sensor and
  observation kernels run outside the graph.
- **Not verified:** one MuJoCo-Warp kernel (`_qfrc_constraint_from_grad`) was traced but is absent from the graph dump (likely inside
  a conditional body the dump does not expand); the diagnostic ran with only 16 environments.

Run from an Isaac Lab 3.0.0 checkout (`export OMNI_KIT_ACCEPT_EULA=YES`; `G1LAB` = path of this repository, see the main README):

```bash
python "$G1LAB"/diagnostics/diag_gpu_physics.py --evidence /tmp/g1_diag \
  --task Isaac-Velocity-Flat-G1 --checkpoint <run>/model_1499.pt --num_envs 16 --train_env_cfg physics=newton_mjwarp
python3 -I "$G1LAB"/diagnostics/analyze_trace.py /tmp/g1_diag
```

(The Chrome trace, 23 MB, and logs are not included. `analyze_trace.py` also needs `traced_kernels.json` and `newton_graph.dot`, which are included.)
