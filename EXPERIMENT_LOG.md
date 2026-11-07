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
