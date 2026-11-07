# G1 Locomotion Lab (integrated, PRIVATE, local only): STATUS
_Updated 2026-11-06 21:25 MST._ Nothing here has been pushed; the public repo `g1-locomotion-lab` is unchanged and must not receive private content without Kaleb's review.

## Goal
Simulated Unitree G1 (Isaac Lab 3.0.0, Newton/MJWarp): walk -> fall -> recover unassisted -> stable stand -> resume walking/turning. Targets: >=50% plausible unassisted recoveries on a fixed held-out set; >=50% recovery->walking chains; stretch: continuous walk->push->fall->recover->walk/turn. These are goals, not results.

## Verified so far
- Walking baseline reproduced under one protocol on both robot models (3 s settle, 10 s at 0.8 m/s, no pushes, 256 envs): minimal-collision `Isaac-Velocity-Flat-G1` 253/256 (6.33 m along heading, tracking err 0.094), full-collision recovery robot 256/256 (6.37 m, 0.093): **no degradation** (`runs/walk_only_*.json`).
- Compatibility (numerically, `runs/compat_*.json`): identical joint order (37), body names, obs terms/order/dims (123), action scale/offset, rates, default pose, joint limits, stiffness/damping/effort limits. Differences: collision shapes 3 vs 38, USD g1_minimal vs g1 (32.96 vs 33.18 kg), episode length, standing-env fraction.
- Evaluator fixed (it used the legacy stand tracker, 1 m Euclidean displacement, no heading/speed/plausibility): `recovery/g1_recovery/metrics.py` + 11 passing tests; protocol v2 frozen in `EVAL_PROTOCOL.md` (DEV/HELDOUT/TEST sets in `recovery/sets/`; position-limit validity added).
- **Recovery:** NO plausible unassisted recovery under protocol v2. The earlier "2/256" (V11) fails v2 (joint position limits violated on 51% of steps); the seed-3 replicate has 0 supported stands on HELDOUT_v2 (assist off or on). A8c's 256/256 and A6's 4/256 chain are rejected (launch exploit).

## Running (Isaac MCP jobs, 16384 envs, 800 its, resumed from Lane A v1 sit-up policy, fresh optimizer, seed 2)
- `20261106-212038-G1-Recovery-R1-v0`: V11 recipe (velocity-limit + launch penalties + gated assist) at 16384 envs.
- `20261106-212040-G1-Recovery-R2-v0`: R1 + body-only joint-position-limit penalty (-10).
- Throughput profile (MCP, 3 its): 16384 envs ~160k transitions/s with two other 4096-env jobs running, 7.7 GB VRAM (4096-env jobs: 41-85k/s each).
- Old v12 jobs (4096 envs) still finishing in `projects/g1-recovery` (V12_cont_assist_1200, V12_cont_noassist_1200).

## Notes
- Isaac MCP jobs load the recovery tasks through `--external_callback g1_recovery.register.register`; this needs `g1_locomotion_lab_dev.pth` in the Lab venv site-packages (adds `recovery/` to sys.path; delete the file to undo). MCP checkpoints must live under the Lab logs dir (init copy: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/init/`). MCP job logs/checkpoints: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/<timestamp>/`.
- Recipes are baked into task ids (no env vars): `G1-Recovery-R1-v0`, `R2-v0` and `-Eval-v0` variants (assist off).

## Next
1. Evaluate R1/R2 on HELDOUT_v2 (assist off/on, >=3 evaluations each) with `recovery/scripts/evalv2.sh`.
2. Depending on position-limit result and supported-stand counts: unactuated settle / action-scale curriculum / more steps / per-family pose training.
3. Stage 2 chain only after a plausible supported stand exists; Stage 3 later.
