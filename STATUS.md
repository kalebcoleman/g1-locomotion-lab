# G1 Locomotion Lab (integrated, PRIVATE, local only): STATUS
_Updated 2026-11-07 04:20 MST._ Nothing here has been pushed; the public repo `g1-locomotion-lab` is unchanged and must not receive private content without Kaleb's review.

## Goal
Simulated Unitree G1 (Isaac Lab 3.0.0, Newton/MJWarp): walk -> fall -> recover unassisted -> stable stand -> resume walking/turning. Targets: >=50% plausible unassisted recoveries on a fixed held-out set; >=50% recovery->walking chains; stretch: continuous walk->push->fall->recover->walk/turn. These are goals, not results.

## Verified so far
- Walking baseline reproduced under one protocol on both robot models (3 s settle, 10 s at 0.8 m/s, no pushes, 256 envs): minimal-collision `Isaac-Velocity-Flat-G1` 253/256 (6.33 m along heading, tracking err 0.094), full-collision recovery robot 256/256 (6.37 m, 0.093): **no degradation** (`runs/walk_only_*.json`).
- Compatibility (numerically, `runs/compat_*.json`): identical joint order (37), body names, obs terms/order/dims (123), action scale/offset, rates, default pose, joint limits, stiffness/damping/effort limits. Differences: collision shapes 3 vs 38, USD g1_minimal vs g1 (32.96 vs 33.18 kg), episode length, standing-env fraction.
- Evaluator fixed (it used the legacy stand tracker, 1 m Euclidean displacement, no heading/speed/plausibility): `recovery/g1_recovery/metrics.py` + 13 passing tests; protocol v2, then v2.1 (hard-limit excursions, headline metric `plausible_v21`), frozen in `EVAL_PROTOCOL.md` (DEV/HELDOUT/TEST sets in `recovery/sets/`; position-limit validity added).
- **Recovery:** NO plausible unassisted recovery under protocol v2. The earlier "2/256" (V11) fails v2 (joint position limits violated on 51% of steps); the seed-3 replicate has 0 supported stands on HELDOUT_v2 (assist off or on). A8c's 256/256 and A6's 4/256 chain are rejected (launch exploit).

## Running (Isaac MCP, 16384 envs, seed 3, no assist; monitor: `python3 recovery/scripts/mon.py 2352 0013 0025 0034 0224`)
- v15, unchanged: `20261106-215641` R6c control; `-225219` R7u; `-225220` R7h; `-225222` R7m; `-225224` R7u warm.
- v16 addition: `20261107-001356` R8w (E5 stance penalty, scratch); `-001359` R8w warm from R7m it500. R6c finished at iteration 1000.
- E4: `20261107-002528` R8t (real torque limits, scratch). Candidate: `20261107-003416` R9h (warm from R7h it650, several changes, direct v3 attempt).
- v17 addition: R9hg (ground-only flight penalty, warm from R9hc it1000).
- v17: `20261107-022451` R9hc and `-022453` R8wc (self-collision on, warm starts). R7 runs and R7u warm finished.
- v16, from the independent review: `20261106-235212` R8 control; `-235213` R8f (E1); `-235215` R8s (E2 supine); `-235216` R8p (E2 prone).
Check rule for v16: DEV at iteration 300 with video, then audit any pass with `diag_trace.py` + `audit_trace.py`. HELDOUT only for the final comparison (3 evaluations each); TEST untouched.
**Best verified result:** R9hg it1100 (`init/` not yet; run `2026-11-07_03-02-51/model_1100.pt`): full v3 on HELDOUT 127-129/256 (49.6-50.4%) over three evaluations, incl. self-penetration on all robots. By family: prone ~78%, side_a ~50%, supine ~46%, side_b ~27%. One seed, warm-start chain, TEST untouched, no walking chain yet.
**Closest behaviour:** R7m and R7h iteration 500 hold a stable wide straddle from all directions (strict 255 and 256 of 256) but score 0 under v2.1, with an ankle roll at its hard stop on 75-83% of steps. Not counted.
**Criteria:** v3 (v2.1 + Amendment 3), approved 2026-11-07. Under v3 every checkpoint so far scores 0. R7h iteration 650 passes the older v2.1 on 174/256 DEV poses but fails v3 on stance and torque.

## Notes
- Isaac MCP jobs load the recovery tasks through `--external_callback g1_recovery.register.register`; this needs `g1_locomotion_lab_dev.pth` in the Lab venv site-packages (adds `recovery/` to sys.path; delete the file to undo). MCP checkpoints must live under the Lab logs dir (init copy: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/init/`). MCP job logs/checkpoints: `envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r/<timestamp>/`.
- Recipes are baked into task ids (no env vars): `G1-Recovery-R1-v0`, `R2-v0` and `-Eval-v0` variants (assist off).

## Documentation
- `recovery/architecture/SYSTEM_OVERVIEW.md`: version names, active jobs and their controls, PPO settings, network architecture, reward weights, limitations, sources.
- `recovery/architecture/figures/recovery_system_dark.png`, `recovery/architecture/figures/network_architecture_dark.png` (sources: `recovery/architecture/*_diagram.html`).

## Next
1. R8w warm: DEV after 100 new iterations; is the stance narrowing, and does it hold a stand?
2. v16 DEV check at iteration 300; compare E1 and E2 with R8, per direction.
3. E3 after stable supported stands; E4 (real torque limits) if approved; two more seeds for any winner.
4. Stage 2 chain only after a stand passes the agreed criteria.
