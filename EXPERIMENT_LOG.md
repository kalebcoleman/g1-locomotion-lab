# Experiment log (hypothesis -> change -> result -> next). Failed and exploited attempts are kept. Pre-integration history: `recovery/STATUS.md`, `recovery/REPORT.md`.
Metric names: legacy = pelvis+upright 2 s; strict = supported stand 2 s; v2 = strict + limits/flight/launch/no assist (`EVAL_PROTOCOL.md`).

## v12 (4096 envs, 1200/600 its; launch penalty; `projects/g1-recovery/lanes/a/runs/V12_*`), HELDOUT_v2, assist off
| run | legacy | strict | v2 | ever supported | note |
|---|---|---|---|---|---|
| V12 cont. of V11 assist (3 evals) | 0,0,0 | 0,0,0 | 0 | 0 | ALL 256 "rose not upright"; no assist force even when enabled (torso never upright enough for the gate) |
| V12 cont. of V11 no-assist (2 evals) | 6,? | 5,2 | 0,0 | 65,? | strict holds violate soft position limits on 89-94% of steps |
| V12 rep seed 3 | 0 | 0 | 0 | 0 | the V11 2/256 did not reproduce |
| V11 (seed 2) | 2 | 2 | 0 | 17 | preliminary 2/256 fails v2 (position limits) |

## Findings driving v13
1. **Curriculum never reached lying starts.** Promote/demote pinned every non-exploit run at mean level <= 1.08 (stand/sit). Only A8c (exploit) reached level 4. Policies were evaluated on poses they had never trained on.
2. **Assist decayed to 0 within ~125 iterations** in R1/R2-style runs (legacy criterion, 0.2 per episode, satisfied by the assist itself). R3 fixes the criterion (supported stand >= 1 s, 0.1 per episode).
3. **Assist gate** requires torso up >= 0.8, so a robot crouched on hands and feet gets no help.
4. **Warm-start sigma = 0.32** => ~0.16 rad joint-target noise; a deep knee tuck needs a mean action near 3.4 at scale 0.5. Exploration reach is a suspect.
5. Standing at wide stance likely violates narrow ankle-roll soft limits (+-0.236 rad).

## v13 (Isaac MCP, 16384 envs, resume Lane A v1 actor+critic, fresh optimizer, fixed lr 3e-4, seed 2)
| job | task | single change | hypothesis |
|---|---|---|---|
| 20261106-212038 | R1 | V11 recipe at 16384 envs | does scale alone help? |
| 20261106-212040 | R2 | R1 + body-only position-limit penalty (-10) | removes limit hugging |
| 20261106-213149 | R3 | R2 + assist decay on supported stand >= 1 s | assist survives long enough to teach |
| 20261106-215302 | R4 | R3 + lying-dominant fixed start mix (5/10/15/30/40 % stand/sit/mid/lying/lying-hard) | policy must train on what it is scored on |
| 20261106-215305 | R4n | R4 without assist | is assist needed once lying starts are trained? |
| 20261106-215307 | R5 | R4 + action scale 1.0 with mean-equivalent checkpoint, sigma kept (joint noise x2) | exploration reach limits the leg tuck |

## Throughput (combined, all jobs at 16384 envs, GPU also shared when noted)
1 job ~160 k/s (with two 4096-env jobs running); 3 jobs ~300 k/s combined (~100 k each, 23.8 GB, 100 % util, 300 W). 6 jobs: to be measured.

## v13 interim (2026-11-06 21:56): R1-R3 cancelled at ~500 its (mine; checkpoints kept), HELDOUT_v2 assist off
| run (checkpoint) | legacy | strict | v2 | ever supported | pos-limit steps | peak vz |
|---|---|---|---|---|---|---|
| R1 @1500 | 0 | 0 | 0 | 0 | 0.29 | 0.73 |
| R2 @1500 | 0 | 0 | 0 | 0 | **0.06** | 0.58 |
| R3 @1300 | 0 | 0 | 0 | 0 | 0.13 | 0.61 |
The body-only position-limit penalty works (0.29 -> 0.06, under the v2 limit of 0.10). None of these train on lying starts (levels 0-1 only); cancelled to free compute for lying-start runs.

## v14 (added 21:56; R4/R4n/R5 from v13 still running; all 16384 envs, 1000 its, fixed lr 3e-4)
Source for the ideas: g1-humanoid-rl ch. 14 (MIT, https://github.com/chaoticcuriosity-io/g1-humanoid-rl): from-scratch PPO with lying-only starts reached a stand with a floor-to-stand height ramp weight 10, upright 2. Not verified against our plausibility criteria; our robot, action space and penalties differ.
| job | task | init | change vs | hypothesis |
|---|---|---|---|---|
| 20261106-215637 | R4n (seed 3) | scratch (sigma 1.0) | R4n warm: no warm start | the sit-up warm start is a local optimum |
| 20261106-215640 | R6b | Lane A v1 | R4n warm: height 10, upright 2 | the height signal is too weak vs penalties |
| 20261106-215641 | R6c (seed 3) | scratch | R6b: lying-only starts, no warm start | closest to the precedent, plus our no-launch/limit penalties |
Selection rule: successive halving on `Curriculum/lying_ever_supported`, `lying_strict_hold`, `lying_max_h` at ~300 iterations, then HELDOUT_v2 v2 evaluation. TEST stays untouched.

## Throughput
1 job: ~160 k/s (while two 4096-env jobs ran). 3 jobs: ~300 k/s combined (~100 k each). 6 jobs: ~43 k/s each (~260-300 k/s combined). Combined throughput saturates near 300 k/s at the GPU's 300 W cap, so more than ~3 jobs only divides it. VRAM ~7.7 GB per 16384-env job.

## Action-scale verification (2026-11-06; requested by Kaleb), `results/actdiag_*.json`, on real simulated DEV-set states
- Mean equivalence of the scale-1.0 checkpoint (`laneA_v1_situp_it999_scale1p0.pt`) vs the original at scale 0.5: joint-target difference max 2.4e-6 rad, mean 3e-7 rad. Confirmed at t=0 only; R5's training then diverges by design.
- Distribution (both, same in rad): target offset |mean| 0.25 rad, p99 1.6 rad. Raw |a| at scale 0.5: mean 0.49, p99 3.21, max 10.56; at scale 1.0: mean 0.25, p99 1.6, max 5.06. Output layer is linear and `clip_actions` is null, so the network itself never saturates; the effective saturation is joint targets beyond limits: 1.7% of joint-steps beyond soft, 1.0% beyond hard, up to 2.66 rad beyond soft (right elbow).
- **Correction:** I had claimed a deep knee tuck was out of reach because it needs a mean action near 3.4 at scale 0.5. Raw actions already reach p99 3.2 and max 10.6, so reach of the MEAN is not the bottleneck. R5 tests only doubled exploration noise (0.16 -> 0.32 rad) plus any learning-dynamics effect of the rescaled output. To re-check on R5's trained checkpoints with `scripts/diag_actions.py`.
- Sit-up policy past HARD limits on 9.5% of steps (any joint), up to 0.22 rad (hip pitch); soft 16.5%. Led to criteria v2.1.
