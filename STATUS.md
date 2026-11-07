# G1 Locomotion Lab (integrated, PRIVATE, local only): STATUS
_Updated 2026-11-06 23:22 MST._ Nothing here has been pushed; the public repo `g1-locomotion-lab` is unchanged and must not receive private content without Kaleb's review.

## Goal
Simulated Unitree G1 (Isaac Lab 3.0.0, Newton/MJWarp): walk -> fall -> recover unassisted -> stable stand -> resume walking/turning. Targets: >=50% plausible unassisted recoveries on a fixed held-out set; >=50% recovery->walking chains; stretch: continuous walk->push->fall->recover->walk/turn. These are goals, not results.

## Verified so far
- Walking baseline reproduced under one protocol on both robot models (3 s settle, 10 s at 0.8 m/s, no pushes, 256 envs): minimal-collision `Isaac-Velocity-Flat-G1` 253/256 (6.33 m along heading, tracking err 0.094), full-collision recovery robot 256/256 (6.37 m, 0.093): **no degradation** (`runs/walk_only_*.json`).
- Compatibility (numerically, `runs/compat_*.json`): identical joint order (37), body names, obs terms/order/dims (123), action scale/offset, rates, default pose, joint limits, stiffness/damping/effort limits. Differences: collision shapes 3 vs 38, USD g1_minimal vs g1 (32.96 vs 33.18 kg), episode length, standing-env fraction.
- Evaluator fixed (it used the legacy stand tracker, 1 m Euclidean displacement, no heading/speed/plausibility): `recovery/g1_recovery/metrics.py` + 13 passing tests; protocol v2, then v2.1 (hard-limit excursions, headline metric `plausible_v21`), frozen in `EVAL_PROTOCOL.md` (DEV/HELDOUT/TEST sets in `recovery/sets/`; position-limit validity added).
- **Recovery:** NO plausible unassisted recovery under protocol v2. The earlier "2/256" (V11) fails v2 (joint position limits violated on 51% of steps); the seed-3 replicate has 0 supported stands on HELDOUT_v2 (assist off or on). A8c's 256/256 and A6's 4/256 chain are rejected (launch exploit).

## Running (Isaac MCP, 16384 envs, 1000 its, lr 3e-4, no assist, lying-only starts, seed 3; monitor: `python3 recovery/scripts/mon.py 215641 2252`)
- `20261106-215641` R6c from scratch (CONTROL; forward-fold optimum at its check).
- `20261106-225219` R7u: orientation reward with a gradient below horizontal.
- `20261106-225220` R7h: height term on imu_link.
- `20261106-225222` R7m: pelvis height gated by torso orientation.
- `20261106-225224` R7u warm-started from the R6c fold (iteration 350).
Check rule: at ~300 its of the from-scratch R7 runs, DEV set, assist off, v2.1 + videos, then prune. HELDOUT only for final comparison; TEST untouched.
Best verified result: **no valid get-up yet.** Closest behaviour: R6c, all poses reach a forward fold (pelvis 0.52 m, torso upside down) within ~1 s.

## Notes
- Isaac MCP jobs load the recovery tasks through `--external_callback g1_recovery.register.register`; this needs `g1_locomotion_lab_dev.pth` in the Lab venv site-packages (adds `recovery/` to sys.path; delete the file to undo). MCP checkpoints must live under the Lab logs dir (init copy: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/init/`). MCP job logs/checkpoints: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/<timestamp>/`.
- Recipes are baked into task ids (no env vars): `G1-Recovery-R1-v0`, `R2-v0` and `-Eval-v0` variants (assist off).

## Documentation
- `docs/SYSTEM_OVERVIEW.md`: version names, active jobs and their controls, PPO settings, network architecture, reward weights, limitations, sources.
- `docs/figures/recovery_system_dark.png`, `docs/figures/network_architecture_dark.png` (sources: `docs/*_diagram.html`).

## Next
1. v15 check when the from-scratch R7 runs reach iteration 300: DEV set, assist off, v2.1, videos, contact sheets and plots; compare each treatment with R6c and separate the warm-start effect from the reward change.
2. If none stands: propose the next controlled experiments from the diagnostics. Queued: head_link COM height reward if R7h stalls.
3. Stage 2 chain only after a plausible supported stand exists; Stage 3 later. HELDOUT for the final comparison; TEST once, after a candidate is selected.
