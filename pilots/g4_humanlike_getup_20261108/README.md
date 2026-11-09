# Goal 4: human-like get-up (training): target not met, a better get-up found

**Target** (set before the run): peak foot spread below 0.8 m with the pelvis above 0.12 m, while the v3 HELDOUT stand rate stays at 90% or more of the baseline `R9hg_it1100`.
**Result: not met.** No checkpoint reduced the peak spread (1.38 m for the baseline, 1.39-1.41 m for every candidate). **A different result came out of the work:** the training setup that makes continued training stable also raises HELDOUT v3 from 129.3 to 174.7 of 256 (+35%) with fewer limit violations.

Training used 6.29 of the 8.00 GPU-hours, counted conservatively as the sum of each run's wall time (several runs shared the GPU, so true GPU time is lower; two runs stopped mid-stage are included). Originals (`R9hg_it1100`, `R9hc_it1000`, `R6cS_fold_it350`, `laneA_v1_situp_it999`) were only read. TEST was not touched; DEV chose checkpoints, HELDOUT was used once per final candidate (three evaluations each).

## Final candidates, HELDOUT (3 evaluations each; v3 before 3b, ground-only flight, assist off)
| checkpoint | HELDOUT v3 before 3b, 3 evals (of 256) | mean | by direction, mean (side_a / prone / side_b / supine) | hard-limit step frac | soft-limit step frac | peak spread, pelvis > 0.12 m | final spread | joint acc RMS | DEV v3 |
|---|---|---|---|---|---|---|---|---|---|
| baseline R9hg it1100 | 125 / 131 / 132 | 129.3 (50.5%) | 31 / 47 / 19 / 33 | 0.0116 | 0.039 | 1.38 m | 0.43 m | 168 rad/s^2 | 92 |
| recA (g4g1 w8p0 it1350) | 170 / 163 / 171 | 168.0 (65.6%) | 53 / 54 / 35 / 26 | 0.0015 | 0.012 | 1.40 m | 0.44 m | 165 rad/s^2 | 148 |
| recB (g4i2 a1 it1421) | 171 / 176 / 168 | 171.7 (67.1%) | 46 / 39 / 46 / 40 | 0.0012 | 0.007 | 1.39 m | 0.48 m | 136 rad/s^2 | 150 |
| recC (g4j1 b1p2 it1425) | 176 / 174 / 174 | 174.7 (68.2%) | 52 / 52 / 36 / 35 | 0.0012 | 0.010 | 1.39 m | 0.42 m | 171 rad/s^2 | 150 |
- Baseline: this table shows its last three HELDOUT evaluations (125 / 131 / 132); earlier ones this run were 123 / 131 / 128, and the historical figure is 127-129.
- 3b (self-penetration): `recA` 238 of 238 and `recC` 236 of 236 robots examined are within the 3% rule (median worst depth 4.5 and 3.4 mm); the sim runs with self-collision on. So full v3 equals the figure above.
- **Best: `recC`**, 174.7 of 256 (68.2%) against 129.3 (50.5%). Hard-limit steps 0.12% (baseline 1.16%), soft-limit steps 1.0% (baseline 3.9%). It was trained with hip roll and yaw targets clipped to +-1.2 rad in the last stage and is evaluated here **without** the clip. The weakest direction is still supine (35 of 64).
- Checkpoint committed: `ckpt/g4j1_b1p2_it1425.pt` (sha256 `a518419e3474...`); `recA` `07d47ddcd32e...`, `recB` `4c6ff0b1423b...` stay local.
- Movement during the rise, all candidates: peak foot spread 1.39-1.40 m at about 0.4 s with the pelvis at 0.33 m (baseline 1.38 m), final spread 0.42-0.48 m, joint-acceleration RMS 136-171 rad/s^2 (baseline 168). Video and contact sheet (first 16 HELDOUT poses, same camera): baseline and `recA` show the same wide straddle at 0.5 s.

## What was found, in order
1. **Continued training of R9hg collapses regardless of the stance penalty.** With a stance weight of only -0.25 the DEV v3 count fell 93 -> 67 -> 24 over 25 / 50 / 75 iterations while the peak spread did not move. Unchanged-reward controls collapsed too, at learning rates from 3e-5 down to 3e-6 and with the entropy bonus off (rows g4a-g4c). Training reward rose to about 106 within 25 iterations while v3 fell and soft-limit violations climbed from 0.04 to 0.37.
2. **A heavy joint-limit penalty stabilises it.** `dof_limits_body` -10 -> -100 with lr 3e-5 kept a no-stance control at DEV v3 93 / 90 / 89 (baseline 95) over 75 iterations (g4d3). With it, further training even improves v3 (DEV up to 158). This recipe is what produced `recA` / `recB` / `recC`.
3. **The stance penalty does not remove the straddle.** First it saturated: `stance_excess` is clamped at 1 beyond `full` (0.85 m), so a 1.3 m flare had no gradient to 0.8 m; with `full` raised to 1.5 m the penalty grows over the whole flare. The peak spread still stayed at 1.40 m at weights up to -8 (g4g, g4h). In training the stance term scaled with the weight and never decreased (about -0.3 per episode at -4, -0.55 at -8): the policy does not trade the flare away, because it is part of the fast rise the height reward asks for.
4. **Smoothness penalties and joint bounds do not either.** Ramping `dof_acc` from -1e-6 up (and `action_rate` from -0.03 up) left the spread at 1.39-1.40 m until v3 collapsed: at `dof_acc` -1e-5 alone, and at -3e-5 with `action_rate` -0.2 (g4i). Bounding hip roll and yaw targets inside training from 1.5 down to 0.5-0.7 rad (g4j) did the same: the policy routes the straddle through other joints. At evaluation time on `recA`, bounding single joint groups moves the peak little (hip roll +-0.5: 1.31 m; hip yaw +-0.5: 1.35 m) and bounding hip pitch breaks standing (v3 39, final spread 1.12 m).
5. **The straddle is the lineage's standing strategy.** Every preserved chain checkpoint that stands (strict stand >= 226 of 256: R7m it500, R7h it650, R8wW it850, R9h it900, R9hc it1000, R9hg it1100) has a peak spread of 1.30-1.42 m; the two with small spread (laneA_v1 sit-up 0.51 m, R6cS fold 0.42 m) never stand (`results/chainprobe/`). At the peak the feet are 1.19 m apart sideways in the pelvis frame and each is one leg length (0.70 m) from the pelvis, with the pelvis at 0.33 m.
6. **Discovery with the penalty from the start did not stand.** From R6cS fold and from laneA sit-up, with stance -1 / -4 / -8 and the stable recipe, neither run stood after more than 850 cumulative iterations (v3 0, strict stand 0, stand reward exactly 0); stopped. A first attempt of these also ended early because of a bug in the driver (it counted a non-standing policy's small spread as meeting the target); fixed, with a unit test.

## All runs (DEV v3 per checkpoint, baseline bar of that run in brackets; peak spread over the same checkpoints)
| run | what | checkpoints evaluated | DEV v3 range (baseline bar in run) | peak spread range (m) | training h |
|---|---|---|---|---|---|
| g4a | lr 3e-4, stance -0.25 early-rise (gate 0.12 m, free 0.45, full 0.85), 75 it | 3 | 24-93 (91) | 1.38-1.39 | 0.09 |
| g4b | lr 3e-5, control (unchanged reward) then stance ramp (stopped after control) | 3 | 22-72 (87) | 1.39-1.39 | 0.08 |
| g4c1 | control, lr 3e-6, entropy 0 | 3 | 10-68 (90) | 1.39-1.39 | 0.17 |
| g4c2 | control, lr 3e-5, entropy 0 | 3 | 22-51 (91) | 1.39-1.39 | 0.17 |
| g4c3 | control, lr 3e-6, entropy 0.008 | 3 | 18-79 (99) | 1.39-1.39 | 0.17 |
| g4d1 | control, lr 3e-6, dof_limits_body -100 | 3 | 67-88 (91) | 1.39-1.40 | 0.16 |
| g4d2 | control, lr 3e-6, dof_limits_body -300 | 3 | 64-83 (95) | 1.40-1.42 | 0.16 |
| g4d3 | control, lr 3e-5, dof_limits_body -100 | 3 | 89-93 (95) | 1.41-1.41 | 0.16 |
| g4f1 | stable recipe + stance ramp -0.25..-8, full 0.85 (stopped) | 5 | 71-105 (93) | 1.41-1.41 | 0.42 |
| g4f2 | same with dof_limits -300 (stopped) | 5 | 55-94 (99) | 1.41-1.42 | 0.42 |
| g4f3 | coarse ramp -1/-3/-8/-15, full 0.85 | 6 | 78-106 (96) | 1.40-1.41 | 0.27 |
| g4g1 | stable recipe + stance ramp -0.5..-8, free 0.45, full 1.5 | 13 | 76-144 (94) | 1.40-1.41 | 0.58 |
| g4g2 | same, free 0.30 | 13 | 70-128 (93) | 1.40-1.41 | 0.58 |
| g4h | from g4g1 w2: stance -8/-16, full 1.5 | 2 | 135-136 (90) | 1.41-1.41 | 0.15 |
| g4i1 | from g4g1 w4: dof_acc ramp -1e-6..-3e-5, stance -8 | 6 | 14-148 (99) | 1.39-1.40 | 0.22 |
| g4i2 | same + action_rate ramp -0.03..-0.2 | 8 | 9-158 (100) | 1.39-1.40 | 0.26 |
| g4j1 | from recA: hip roll/yaw bounds 1.5..0.5 rad in training | 10 | 57-155 (94) | 1.39-1.40 | 0.35 |
| g4j2 | same + hip pitch +-1.6 | 8 | 87-135 (101) | 1.40-1.41 | 0.31 |
| g4m1 | discovery from R6cS_fold_it350, stance -1/-4/-8 | 5 | 0-0 (99) | 0.44-0.52 | 0.34 |
| g4m1b | continuation of g4m1 | 6 | 0-0 (94) | 0.41-0.47 | 0.44 |
| g4m2 | discovery from laneA_v1_situp_it999 | 5 | 0-0 (99) | 0.40-0.50 | 0.34 |
| g4m2b | continuation of g4m2 | 6 | 0-0 (94) | 0.43-0.46 | 0.44 |
DEV v3 of the baseline varies between 86 and 104 across evaluations, so the bar (90% of the baseline measured at the start of each run) moves a little from run to run.

## Recommended next step
The straddle is robust to every reward-side and bound-side change tried here. A policy that is both human-like and standing has to be discovered differently: imitation of a human get-up clip (the roadmap's gated item; `REFERENCE_DATA_RESEARCH.md`: LAFAN1 licence and the one-waist-joint morphology are the blockers), or a curriculum that teaches the stand from a sitting start (the sit-up policy has a 0.5 m spread) before lying starts. Independently of style, adopt `recC` as the new get-up baseline (v3 68%): the stable recipe is a real improvement and costs about 10 minutes of GPU per 75 iterations.

## Limits
- One training lineage; single seeds per run (the plan seeds are in `plan_*.json`); DEV selection of checkpoints from many evaluations, so DEV maxima are optimistic, which is why the table above uses HELDOUT, never used for selection. HELDOUT figures are three evaluations of one checkpoint each; simulation is not bit-repeatable.
- The peak-spread metric is the mean over robots of the largest ankle-to-ankle distance while the pelvis is above 0.12 m; "human-like" itself is not measured by any metric here, and the straddle is the only style descriptor tested.
- `recC` was trained with a hip roll/yaw clip and evaluated without it; behaviour under that clip was not separately evaluated.
- `driver_g4.py` was refactored after the runs so that the selection rule (`selection.py`, 4 CPU tests) is testable; the rule is unchanged.

## Reproduce
```
pilots/g4_humanlike_getup_20261108/train_stage.sh <run_name> 16384 <iters> <seed> <init.pt> <hydra overrides...>
python pilots/g4_humanlike_getup_20261108/driver_g4.py pilots/g4_humanlike_getup_20261108/plan_g4g1.json     # stages, DEV evaluation, selection
pilots/g4_humanlike_getup_20261108/final_eval.sh <candidate.pt> <tag>      # HELDOUT x3 and DEV for baseline and candidate, videos
python pilots/g4_humanlike_getup_20261108/summarize_final.py <tag>
```
Plans store the checkpoint paths as `<workspace>/...`. Stable-recipe overrides: `env.rewards.dof_limits_body.weight=-100.0 agent.algorithm.learning_rate=3e-5` (plus the stance overrides `env.rewards.stance.weight`, `env.rewards.stance.params.{pelvis_h,free,full}`).
