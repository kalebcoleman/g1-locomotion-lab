# M2b: hand-over fixes for the frozen walk -> fall -> get-up -> walk sequence (evaluation only)

No training and no weight changes: the walking policy (`walking_newton_it1499.pt`) and the recovery policy (`R9hg_it1100`) are used as-is. Same runner, push magnitudes (0, 2.5, 3.5, 4.5 m/s at 3.0 s), 128 envs per seed, DEV-style (no pose sets; TEST untouched). Baseline from M2: 6 full successes in 250 induced falls under the v3 gate, 61 of 98 hand-overs re-fell.

## What was added (`recovery/g1_recovery/fall_handover.py`, `eval_fall_handover.py`, 24 CPU tests)
- **Settle**: after the trigger, the recovery policy (or a hold of the current joint positions) keeps acting for N steps before the switch; the trigger is cancelled if the robot stops being stable.
- **Posture blend**: during the last steps of the settle the action is ramped to the default stance, for all joints, for the fingers only, or for the "upper" joints (shoulders, elbows, fingers).
- **Command ramp**: after the switch the forward command ramps from 0 to 0.8 m/s.
- **Readiness gate** (`--trigger ready`): supported stance, pelvis >= 0.60 m, upright >= 0.90, pelvis speed < 0.2 m/s, angular speed < 0.5 rad/s, held 1 s. This is a **different check from the official v3 stand result**, which is still computed and is the headline for the v3 trigger. The readiness gate has no validity clauses (flight, limits, torque).
- Diagnosis of the hand-over state against the no-push walking control (below).

## Results, seeds 2-4 (run 1; numbers summed over the 3 seeds)
Full success = hand-over, then 10 s of walking with no re-fall, >= 3 m along the heading and >= 0.5 m/s (the M2 rule). The baseline row reproduces M2 within its run-to-run variance.
| variant | trigger | induced falls | hand-overs | re-fell | mean time to re-fall | walked 10 s | full success | rate | per seed (2, 3, 4) |
|---|---|---|---|---|---|---|---|---|---|
| base: baseline (M2 behaviour) | v3 | 247 | 75 | 41 | 1.67 s | 34 | 2 | 0.8% | [1, 1, 0] |
| settle25: settle 25 steps (0.5 s) | v3 | 251 | 80 | 52 | 1.65 s | 28 | 5 | 2.0% | [1, 1, 3] |
| settle50: settle 50 (1 s) | v3 | 250 | 81 | 39 | 1.63 s | 42 | 5 | 2.0% | [1, 1, 3] |
| settle100: settle 100 (2 s) | v3 | 250 | 62 | 39 | 1.65 s | 23 | 4 | 1.6% | [2, 2, 0] |
| hold50: settle 50, hold joint positions | v3 | 253 | 0 | 0 | - | 0 | 0 | 0.0% | [0, 0, 0] |
| blendall50: settle 50 + blend all joints to default (50 steps) | v3 | 248 | 0 | 0 | - | 0 | 0 | 0.0% | [0, 0, 0] |
| blendhands50: settle 50 + blend fingers (50) | v3 | 250 | 83 | 0 | - | 83 | 25 | 10.0% | [12, 5, 8] |
| blendupper50: settle 50 + blend arms and fingers (50) | v3 | 248 | 73 | 0 | - | 73 | 73 | 29.4% | [22, 31, 20] |
| blendupper25: settle 25 + blend arms and fingers (25) | v3 | 251 | 68 | 0 | - | 68 | 68 | 27.1% | [25, 24, 19] |
| ramp50: command ramp 50 steps (1 s) | v3 | 250 | 91 | 37 | 1.69 s | 54 | 1 | 0.4% | [0, 0, 1] |
| ramp100: command ramp 100 (2 s) | v3 | 251 | 95 | 33 | 1.6 s | 62 | 0 | 0.0% | [0, 0, 0] |
| ready: readiness trigger | readiness | 249 | 249 | 155 | 1.52 s | 94 | 38 | 15.3% | [7, 12, 19] |
| ready_settle50: readiness trigger + settle 50 | readiness | 251 | 251 | 164 | 1.51 s | 87 | 19 | 7.6% | [7, 7, 5] |
| ready_upper50: readiness trigger + settle 50 + blend arms and fingers (50) | readiness | 251 | 251 | 0 | - | 251 | 251 | 100.0% | [86, 83, 82] |
| upper50_ramp50: v3 trigger + settle/blend upper 50 + ramp 50 | v3 | 248 | 73 | 0 | - | 73 | 71 | 28.6% | [23, 22, 26] |
| ready_upper50_ramp50: readiness trigger + settle/blend upper 50 + ramp 50 | readiness | 251 | 251 | 1 | 0.7 s | 250 | 249 | 99.2% | [84, 82, 83] |
- `hold50` and `blendall50` never reach a hand-over: the robot is not stable while holding joint positions or while every joint is blended to the default stance.
- Settling and ramping alone do nothing useful (0.0-2.0% success). Blending **only the fingers** removes the re-falls (0 of 83) but leaves most walks too slow for the speed rule (25 successes). Blending **arms and fingers** removes the re-falls and the slow walks: every hand-over completes the walk (73 of 73).
- With the readiness gate instead of the v3 gate every fallen robot that stands is handed over (251 of 251) and, with the arm and finger blend, completes the walk (251 of 251). Without the blend the readiness gate alone still re-falls 155 of 249 times.

## Run-to-run variance and fresh seeds
A repeat is an independent run of the identical configuration (simulation is not bit-repeatable). Seeds 5-7 were not used for choosing anything.
| variant | run | seeds | induced falls | hand-overs | re-fell | full success | rate | per seed |
|---|---|---|---|---|---|---|---|---|
| base | 1 | 2-4 | 247 | 75 | 41 | 2 | 0.8% | [1, 1, 0] |
| base | 2 | 2-4 | 251 | 88 | 46 | 6 | 2.4% | [2, 0, 4] |
| base | 3 | 2-4 | 248 | 85 | 45 | 11 | 4.4% | [5, 4, 2] |
| base | pooled | 2-4 | 746 | 248 | 132 | 19 | 2.5% | |
| base | fresh seeds | 5-7 | 243 | 84 | 51 | 2 | 0.8% | [2, 0, 0] |
| blendupper50 (v3 trigger) | 1 | 2-4 | 248 | 73 | 0 | 73 | 29.4% | [22, 31, 20] |
| blendupper50 (v3 trigger) | 2 | 2-4 | 252 | 60 | 0 | 60 | 23.8% | [20, 21, 19] |
| blendupper50 (v3 trigger) | 3 | 2-4 | 249 | 82 | 0 | 82 | 32.9% | [25, 29, 28] |
| blendupper50 (v3 trigger) | pooled | 2-4 | 749 | 215 | 0 | 215 | 28.7% | |
| blendupper50 (v3 trigger) | fresh seeds | 5-7 | 245 | 69 | 0 | 69 | 28.2% | [23, 22, 24] |
| ready_upper50 (readiness trigger) | 1 | 2-4 | 251 | 251 | 0 | 251 | 100.0% | [86, 83, 82] |
| ready_upper50 (readiness trigger) | 2 | 2-4 | 252 | 252 | 0 | 252 | 100.0% | [85, 84, 83] |
| ready_upper50 (readiness trigger) | 3 | 2-4 | 251 | 250 | 0 | 250 | 99.6% | [83, 83, 84] |
| ready_upper50 (readiness trigger) | pooled | 2-4 | 754 | 753 | 0 | 753 | 99.9% | |
| ready_upper50 (readiness trigger) | fresh seeds | 5-7 | 245 | 244 | 0 | 244 | 99.6% | [81, 82, 81] |
## Target check (goal: >= 25% full success under v3 on DEV, re-falls after hand-over <= about 30 of 98, consistent over seeds 2-4)
- **v3-gated best, `blendupper50`**: 28.7% pooled over three repeats (215 of 749) and 28.2% on fresh seeds; **0 re-falls in every run** (baseline 132 of 248 hand-overs over its three runs). The target is met on average but is borderline per run: one repeat gave 23.8%, and the lowest single seed was 19 of about 84 (about 23%). The limit is the number of v3 hand-overs (60-82 of about 250), not the walking after them.
- **Readiness-gated best, `ready_upper50`**: 99.9% pooled over three repeats and 99.6% on fresh seeds. It is not a v3 result: at the switch the official v3 gate was true for only 1-3 of 245-252 robots, and this cannot be split into its causes here: the readiness gate can hand over before the 2 s v3 hold completes (probable, not measured), and for many robots the v3 validity clauses (ground flight, hard-limit excursions) would not pass at all (M2). These robots do stand and walk (M2: 152 timed-out recoveries all reached a supported 2 s stand).
- Post-hand-over walking distance over 10 s: baseline 3.77 m (34 completed windows), `blendupper50` 6.20 m, `ready_upper50` 6.19 m; the no-push control walking from rest covers 6.16-6.19 m. Controls: 96 of 96 still walking in every run.

## Diagnosis of the re-falls (hand-over state vs the walking policy's usual state)
The policy observation at the hand-over step was compared with the no-push control's observations over the 2 s before the push (z-scores, std floored at 0.02). Baseline, 75 hand-overs, seeds 2-4:
| joint group | mean abs position difference from the control mean | control std |
|---|---|---|
| legs | 0.27 rad | 0.07 rad |
| waist (`torso_joint`) | 1.24 rad | 0.07 rad |
| arms | 1.34 rad | 0.03 rad |
| fingers | 1.29 rad | 0.03 rad |
Foot spread at hand-over 0.44 m vs 0.24 m in the control. Mean |z| per observation term: finger joint positions 54, body joint positions 27, last actions 7 (body) and 18 (fingers), base linear velocity 3.9, gravity 3.2, joint velocities below 1.
After the arm and finger blend: arms 0.28 rad, fingers 0.35 rad, last-action z falls to 1.4 / 2.4, joint-position z (body) 27 -> 8, but the waist (1.07 rad) and the wide stance (0.43 m) are unchanged, and they walk fine. So the upper body being far outside the walking policy's input distribution is what causes the re-falls; leg posture, waist yaw and stance width are tolerated. No single dimension separated re-fell from completed robots consistently across seeds (|Cohen d| up to 1.9, different dimensions per seed).

## Do the eval-only fixes plateau? A fine-tune?
No plateau: the target is met with an evaluation-only change, so **no fine-tune was run and no training budget was spent on this goal.** The remaining lever for the official v3 hand-over share (about 29%) is the get-up itself (ground flight and joint-limit excursions during the rise), which is goal 4, not the walking policy. If a walking fine-tune were ever wanted for hand-over robustness (waist yaw, wide stance), the smallest bounded version would be: warm start from `walking_newton_it1499.pt`, learning rate <= 3e-5, 100 iterations, starts mixed from get-up hand-over states (readiness gate) and normal starts, checkpoint every 25 iterations, about 5 min of GPU at 16384 envs, with the no-push control (96 of 96, 6.16 m) as a regression gate. Not run.

## Limits
- Three repeats per variant are needed to see the spread; the sweep itself is one run per variant, so small differences between variants (for example 5 vs 4 successes) mean nothing.
- The blend is an evaluation-time action override (default stance for the upper joints); it is not part of either policy and a real deployment would need it implemented as a hand-over routine.
- The readiness gate is not the official stand result; it was introduced to separate "can walk after standing" from "got up with valid movement". The v3 headline numbers are the `v3` trigger rows.
- The video is one 16-robot run of seed 3 per variant (4 no-push controls, 4 each at 2.5 / 3.5 / 4.5 m/s), uncut, same seed for all three. Baseline: 0 successes (3 re-falls, 8 recoveries that stood but never passed the v3 gate, 1 push without a fall). `blendupper50`: 5 successes, 6 stood but not passed. `ready_upper50`: 11 of 11 fallen robots recovered and walked. Runs are not bit-repeatable, so the video's counts differ slightly from the batch.

## Reproduce
```
./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint <R9hg_it1100.pt> --walk_checkpoint <walking_newton_it1499.pt> --num_envs 128 --seed 2 --levels 0,2.5,3.5,4.5 \
    --settle 50 --blend 50 --blend_joints upper --out <out.json> physics=newton_mjwarp               # v3-gated best
    # readiness-gated best: add --trigger ready;  baseline: no variant flags
pilots/m2b_handover_fixes_20261108/run_variants.sh "2 3 4" "" base blendupper50 ready_upper50     # sweep helper (variants are listed in the script)
python pilots/m2b_handover_fixes_20261108/aggregate_variants.py [--detail] [--seeds 5,6,7] base blendupper50 ready_upper50
```
`ledger.csv`: wall seconds per job (about 30 s each when the GPU is not shared).
