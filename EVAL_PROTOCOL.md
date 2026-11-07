# Evaluation protocol v2.1 (v2 frozen 2026-11-06 21:25 MST, before further optimisation). Supersedes `recovery/CRITERIA.md` v1 + Amendment 1 for NEW results; all earlier results keep their original definitions (v1) and are never re-labelled.
Simulator: Isaac Lab 3.0.0, `physics=newton_mjwarp`, 50 Hz, full-collision G1 (`G1_CFG`), no pushes in recovery evaluation, observation noise/randomisation as in the task config, seed 1 for env construction.

## Pose sets (`recovery/sets/`)
| set | file | seed | composition | purpose |
|---|---|---|---|---|
| DEV | `dev_v1.npz` (= original `heldout_poses.npz`) | 20261106 | 256 lying poses, 4 directions x 64, limb noise 0.4 rad | **Already used throughout iteration: now a development set, not held-out.** |
| HELDOUT | `heldout_v2.npz` | 20261107 | 256 poses = 4 directions (supine, prone, side_a, side_b) x 2 limb-noise levels (0.1 easy, 0.4 standard) x 32, tilt 80-90 deg, shuffled | fixed comparison set; not used for training/selection |
| TEST | `test_v2.npz` | 20261108 | same structure, different seed | **untouched until a final claim**; a run may be scored on it once per final candidate |
Reports always split by direction and by difficulty; a headline rate is never reported without the per-difficulty split.

## Stage 1: plausible unassisted recovery (`g1_recovery/metrics.py: supported_stand, plausible_recovery`)
One 10 s episode from the pose, no reset/teleport after t=0. All must hold:
1. Supported stand >= 2.0 s continuous: pelvis height >= 0.60 m, torso up-vector.z >= 0.90 (tilt <= 25.8 deg), BOTH ankle-roll links with contact force > 10 N and < 0.15 m above floor, **3D** pelvis (root COM) speed < 0.6 m/s (v1 used the horizontal components only).
2. Joint-velocity validity: any body joint above its soft velocity limit on <= 2% of steps.
3. No launch: peak pelvis upward speed <= 2.0 m/s; free flight (no body in contact with anything) on <= 2% of steps.
4. **Joint-position validity (new in v2):** body joints beyond their soft position limits on <= 10% of steps. (Hand joints excluded: their default pose lies outside their soft limits.)
5. No assistance: the assist hook applied zero force during the episode (`assist.force_steps == 0`); assist-on numbers are reported separately and never as the headline.
Always reported next to it: legacy count (pelvis+upright only), strict count (supported stand only), recovery time (start of the successful hold), longest/total supported-stand time, per-direction/difficulty rates, failure causes, contact forces, actuator effort.

**Amendment v2.1 (frozen 2026-11-06, before ANY v2-valid recovery existed, so it cannot favour a result): hard-limit excursions.** Limits in this solver are compliant, so joints can pass them; meeting the soft-limit rule (item 4) does not mean no violation. Added item 6: any body joint past its HARD limit on <= 3% of steps AND max excursion past a hard limit <= 0.10 rad. Anchored to a measured reference: the walking policy (21 s settle+walk+turn, 256 envs, `results/walk_ref_limits.json`) has 1.5% of steps, excess p99 0.083 rad, max 0.105 rad; the Lane A v1 sit-up policy has 9.5% and up to 0.22 rad and must fail. **Headline metric from now on: `plausible_v21`.** v2 is still reported alongside. Earlier results keep their definitions.

## Stage 2: chain, recovery -> walking (`scripts/eval_chain.py`, `metrics.chain_success`)
Episode 20 s. The recovery policy acts until it has held a SUPPORTED stand for the hand-over rule (default **100 consecutive supported steps = 2 s**; shorter rules are exploratory and labelled). One-way switch to the existing walking policy (published run, iteration 1499) with command vx = 0.8 m/s (yaw 0). No switch back. Success requires:
- plausible recovery at hand-over (Stage 1 items 2-5 evaluated on the episode so far; assist inactive),
- the **10 s walking window** completed inside the episode (otherwise `timeout`),
- no fall in the window (pelvis < 0.45 m or torso up < 0.5),
- **>= 3.0 m along the commanded heading** (heading = pelvis x-axis projected on the floor at hand-over; lateral motion does not count),
- mean speed along heading over the window >= 0.5 m/s (note: this means >= 5.0 m in 10 s, so the 3 m rule is implied; both are scored and reported with separate failure reasons).
Failure reasons: never_handed_over, implausible_recovery, timeout_window_incomplete, fell_after_handover, insufficient_progress_along_heading, too_slow. Also reported: command tracking error, falls after hand-over, hand-over time.
The earlier evaluator (v1) used the legacy stand tracker, 1 m Euclidean displacement measured at episode end and no speed/plausibility/heading rules; its results (e.g. the A6 "4/256 chain") are retracted, not re-scored.

## Stage 3: continuous walk -> disturbance -> fall -> recover -> walk/turn (stretch)
One uncut episode, physics always active, no reset/teleport/hidden support; disturbance = impulsive base velocity set through the task's push mechanism (magnitude/direction logged, fixed list in advance, defined when this stage is implemented), fall = pelvis < 0.45 m or tilt > 60 deg; then Stage 1 plausibility from the first ground contact, supported stand, resumed walking >= 3 m and a >= 90 deg yaw turn within 8 s. Switching rules must be fixed functions of observable state.

## Statistics
Each final candidate: >= 3 independent evaluations (the evaluation is not bit-repeatable: one checkpoint scored 3-8/256) on HELDOUT; robustness claims need >= 3 independently trained seeds; report mean and range, attempts and successes. Training budgets for method comparisons are matched in transitions; wall time/throughput are reported separately (shared GPU).

## Tests
`recovery/tests/test_metrics.py` (12 tests: both-feet requirement, airborne lunge, 3D speed, threshold edges, plausibility flags, position-limit violation, heading projection, chain edges/reasons) must pass before any result is trusted.
