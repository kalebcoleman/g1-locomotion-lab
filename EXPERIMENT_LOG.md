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

## Check at iteration 300 (from-scratch runs; 2026-11-06 ~22:40), DEV set, assist OFF, criteria v2.1, plus video review (`media/20261106_224419_chk300_*`)
| run | ckpt | legacy | strict | v2 | v2.1 | ever supported | never rose | rose not upright | peak vz | pos-limit steps | past-hard steps | behaviour on video |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R4 | 1300 | 0 | 0 | 0 | 0 | 0 | 168 | 88 | 0.63 | 0.00 | 0.0% | sits |
| R4n | 1300 | 0 | 0 | 0 | 0 | 0 | 162 | 94 | 0.76 | 0.00 | 0.0% | sits |
| R5 | 1300 | 0 | 0 | 0 | 0 | 0 | 157 | 99 | 0.84 | 0.02 | 0.0% | sits (frames not inspected individually) |
| R4n scratch | 300 | 0 | 0 | 0 | 0 | 0 | 109 | 147 | 0.65 | 0.01 | 0.0% | sits, torso upright |
| R6b | 1300 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 1.13 | 0.05 | 0.2% | sprawled/folded, pelvis raised |
| R6c scratch | 350 | 0 | 0 | 0 | 0 | 0 | 0 | 256 | 1.24 | 0.40 | 3.3% | **all robots in a forward fold within ~1 s, then hold it** |
**No valid get-up.** Diagnosis of R6c (`runs/probeH_V.log`): pelvis 0.52 m, imu_link 0.38 m, feet near the floor, pelvis up-vector **-0.97** (torso upside down). `upright = clamp(up, 0, 1)` is exactly 0 there: no gradient to uncurl. head_link/torso_link report the pelvis origin in this USD (`runs/probeH_stand.log`), imu_link does not (0.846 m standing).
R4 vs R4n: assist made no visible difference at this stage (assist level stayed 1.0, no supported stands). R6 vs R4: the height-dominant reward moved the failure from "sits" to "fold": every pose rises above 0.35 m, none is upright.
**Pruned (mine; checkpoints kept):** R4, R4n, R5, R4n-scratch, R6b. **Kept as control:** R6c scratch.

## v15 (launched ~22:50; all from scratch unless noted, lying-only starts, seed 3, 16384 envs, 1000 its, lr 3e-4, no assist; single change vs R6c)
| task | change | hypothesis |
|---|---|---|
| R7u | upright = (1 + up) / 2, gradient below horizontal | the clamp removes the only signal to uncurl |
| R7h | height term on imu_link instead of pelvis (0.85 m standing) | a fold should not pay the height reward |
| R7m | pelvis height x (0.25 + 0.75 upright_full) (HoST: orientation x height) | same, multiplicatively |
| R7u warm from R6c@350 (`init/R6cS_fold_it350.pt`) | R7u, starting from the fold | can the fold be a stepping stone? |

**Documentation (23:22):** added `docs/SYSTEM_OVERVIEW.md` and dark diagrams in `docs/figures/`. No run was changed or restarted.

## v15 check (2026-11-06 23:28 onward), DEV, assist off, criteria v2.1, uncut video review (`media/20261106_235_chkv15_*`, `results/chk_v15_*`)
Columns: v2.1 passes, strict 2 s supported stands, soft-limit %, hard-limit %, peak pelvis upward speed mean/max (m/s), then robots reaching seated, fold, crouch on feet, high on feet, any supported stand, supported 1 s.
| checkpoint | v2.1 | 2 s | soft | hard | peak vz | seated | fold | crouch | high | any | 1 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R6c it300 | 0 | 0 | 22.1 | 0.84 | 1.21 / 2.89 | 0 | 256 | 0 | 0 | 0 | 0 |
| R6c it650 | 0 | 0 | 28.2 | 0.63 | 1.38 / 2.65 | 0 | 256 | 0 | 0 | 0 | 0 |
| R6c it750 | 0 | 0 | 0.3 | 0.04 | 1.39 / 2.73 | 0 | 256 | 0 | 0 | 0 | 0 |
| R7u it300 | 0 | 0 | 41.6 | 7.44 | 1.28 / 2.68 | 0 | 256 | 0 | 0 | 0 | 0 |
| R7u warm it650 (300 new) | 0 | 0 | 56.0 | 1.16 | 1.33 / 2.68 | 0 | 256 | 0 | 0 | 0 | 0 |
| R7h it300 | 0 | 0 | 19.2 | 2.44 | 1.08 / 2.36 | 27 | 256 | 25 | 20 | 0 | 0 |
| R7m it300 | 0 | 0 | 10.5 | 2.06 | 1.02 / 2.10 | 247 | 256 | 240 | 157 | 0 | 0 |
| R7h it350 | 0 | 0 | 42.7 | 16.53 | 1.17 / 2.27 | 167 | 256 | 165 | 163 | 6 | 0 |
| R7h it400 | 0 | 0 | 63.1 | 37.93 | 1.24 / 2.38 | 256 | 242 | 256 | 256 | 162 | 1 |
| R7m it350 | 29 | 76 | 5.9 | 1.86 | 0.93 / 2.14 | 247 | 191 | 231 | 190 | 155 | 134 |
| R7m it400 | 0 | 2 | 35.8 | 15.02 | 0.88 / 2.10 | 253 | 208 | 221 | 147 | 147 | 103 |
Reward effect (R7x vs R6c, same seed, envs, PPO and starts): R7u did not change posture; R7h and R7m did, R7m most by iteration 300. Each change also alters reward magnitude against the fixed penalties (below), so this isn't a pure signal-shape test. One seed each.
Warm start: R7u warm stayed in the fold, same as R7u from scratch after 300 new iterations; more soft-limit steps (56% vs 42%), fewer hard-limit steps (1.2% vs 7.4%). It's a separate arm, not a matched comparison: its critic trained on the R6 reward and it carries 350 extra iterations.
Reward magnitude per episode (latest logged, R6c it960, R7u it501, R7h it501, R7m it502, R7u warm it849): height 8.69, 8.22, 9.27, 8.36, 8.43; upright 0.00, 0.32, 1.81, 1.75, 0.17; stand 0, 0, 2.58, 2.41, 0; action rate -0.51, -0.59, -1.08, -1.28, -0.47; position limits -0.23, -0.11, -0.23, -0.26, -0.07; flight -0.13, -0.16, -0.41, -0.58, -0.10.

**Audit of R7m it350 (`recovery/scripts/diag_trace.py`, `audit_trace.py`, `results/audit_R7m_it350.json`). NOT counted as a recovery.** A retest scored 21 v2.1 and 73 strict (first run 29 and 76). Of the 21: all weight on the feet during the hold (about 200 N + 146 N; hands, arms, knees, trunk 0 N), no assist, no flight or launch breach. But the feet are 0.87-0.90 m apart during the hold, after a splay to about 1.3 m within 0.5 s; pelvis about 0.62 m against 0.69 m for a normal stand; longest supported run 2.0-3.7 s; 18 of 21 no longer supported at the end, most dropping to a kneel at about 8 s (pelvis 0.44 m, feet 0.37 m apart); peak torque 3.2-6.1x Unitree's limits, above them on 5.8-11.8% of steps, so 0 of 21 meet the proposed 3a. Supine: 0 of 64 in both evaluations. Iteration 400 fell to 0 passes, so the behaviour isn't stable across training either.
Training stat: `lying_strict_hold` stays 0 while deterministic evaluation finds 2 s holds, consistent with exploration noise (std about 0.9) preventing holds in training; this supports the reviewer's E3 once supported stands are stable.

## v16 (launched 2026-11-06 23:52; seed 3, 16384 envs, 1000 its, scratch), plan from Kaleb's independent review; R7* = R7m
Reviewer claims confirmed in code before acting: R1-R7 stand bonus uses the pelvis + tilt mask (`G1R_STRICT` unset); `G1_CFG` legs and arms have a 300 Nm limit and self-collision is off. Also found: this model's ankles are capped at 20 Nm, below Unitree's 50 Nm. Full review file was not readable at the given path; worked from the text pasted in the message.
| job | task | change | control |
|---|---|---|---|
| `20261106-235212` | R8 | R7m + stand bonus on the feet-supported mask | running R7m |
| `20261106-235213` | R8f (E1) | R8 + feet-load share (+3) + prop force above 0.40 m pelvis (-2) | R8 |
| `20261106-235215` | R8s (E2) | R8 + supine family (phi 270 +- 30 deg) + 30-step passive settle | R8 on supine DEV poses |
| `20261106-235216` | R8p (E2) | R8 + prone family (phi 90 +- 30 deg) + 30-step passive settle | R8 on prone DEV poses |
Deviations from the review: control is R8 (with the stand fix), not an exact R7m replicate; the running R7m serves as R8's control. E2 changes two things together, as specified. Settle sets the target to the measured joint position instead of zero action, because zero action would drive stiffly to the default pose; target updates at 50 Hz, so stiffness isn't fully off (mean of per-robot largest joint torque over the 30 steps: 11 Nm, against 32-68 Nm afterwards).
Verified on simulator states (`recovery/scripts/verify_r8.py`, `runs/verify_R8*.log`): family sampling (gravity along body x -0.94 supine, +0.94 prone), feet share 0.997 during supported stands, stand bonus 0 when unsupported, prop force 0 during those stands.
Queued: E3 exploration schedule after stable supported stands. Proposed, not started: E4 train with Unitree's real torque limits, because nothing can pass 3a at scale while the model allows 300 Nm; E5 a stance-width term if the straddle returns.
