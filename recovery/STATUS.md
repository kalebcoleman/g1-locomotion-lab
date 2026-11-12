# G1 recovery (get-up) project: STATUS

> **Historical snapshot (2026-11-06).** For the current project status see [the root STATUS.md](../STATUS.md).
_Last updated: 2026-11-06 18:45 MST_

## Goal
Train and compare two fall-recovery (get-up) approaches for the simulated Unitree G1 (Isaac Lab 3.0.0, Newton/MJWarp), in parallel on the shared GPU:
- **Lane A**: task-reward policy with a progressive fallen-pose curriculum (rewards for upright + stable standing).
- **Lane B**: reference-guided if a suitable, licensed get-up reference exists; otherwise a clearly different curriculum/reward design.
Then pick the stronger, chain it with the existing walking policy (fall -> get up -> stand -> walk), save screenshots/videos of successes and failures.
Original projects (`projects/g1-tasks`, `projects/g1-locomotion-lab`) are read-only inputs. Nothing was pushed or published at this point (rule: branches and pull requests only, never main).

## Resources (16:45, before any of our jobs)
GPU RTX PRO 6000 Blackwell 98 GB: 1.2 GB used, 1% util (Jenga diag just starting). CPU 32 cores, load ~3-5. RAM 125 GB (112 free). Disk 1.5 TB free.

## Findings so far
- `G1_MINIMAL_CFG` (used by walking task) has only 3 collision shapes per robot (2 feet boxes + torso box). Limbs/head/hands cannot touch the ground, so it is unusable for get-up. Recovery uses `G1_CFG` (full collision `g1.usd`, same joints/actuators/limits). Documented difference vs walking.
- No existing get-up task or G1 reference motion exists locally or in Isaac Lab 3.0.0 tasks.

## Current task
Lane A run #1 training (1024 envs x 1000 iters, seed 1). Lane B implementation writing `g1_recovery/lane_b.py` (HoST-style assist-force curriculum + staged rewards; research found no licensed G1 get-up reference, see `research/lane_b_references.md`).

## Done
- Shared base package `g1_recovery/` (env, held-out poses `heldout_poses.npz`, success tracker), `scripts/{train,eval,record}_recovery.py`, `scripts/gpu_run.py` (lead's GPU launch coordinator: flock + free-memory check + memory sampling -> logs/runs.jsonl). Smoke tests of Lane A (train 256 envs ~13.9k steps/s; eval 256 held-out poses; 16-env video) pass.
- Held-out pose rendering checked: `media/20261106_165529_heldout_poses_t0.3s.png` (16 lying robots: supine/prone/sides, resting on floor, no explosions).
- Resource inspection; collision-shape inspection (`logs/inspect_shapes.log`).
- Lane B literature review started (writes `research/lane_b_references.md`).

## External reference check (18:20)
- APEX-humanoid `Isaac-StandUp-G1` / `standup.pt` assessed statically: `research/apex_standup_assessment.md` (PhysX Isaac Lab 1.3.0 fork, 29-DoF modified G1 USD, 558-d obs with 6-step history, 23-d actions, trained only from a prone tucked start; no authors/CITATION, checkpoint/USD have no separate licence). Not directly evaluable on our Newton env; port plan and risks written. No runs started.

## HoST / HumanUP code-level comparison (18:55)
- `research/host_humanup_vs_ours.md`: key gaps = budget (HoST 12k its / HumanUP 50k vs our <=1k), assist gating (HoST: torso-upright gated, per-env decay), 0.6 s unactuated settle, action-scale curriculum, single posture family first, 6-step obs history, multi-critic groups, self-tracking stage II idea. v7 plan written there; not started.

## Stuck-transition diagnostic (A v1 final policy, level-1 sitting starts, 256 envs, 6 s) -- `lanes/a/diag_A1_level1.json`
Pelvis stays 0.17-0.19 m (none ever above 0.35 m), torso upright 0.97 (sits upright), feet in contact 75% of the time (mean 45 N), hands ~107 N and thighs/legs ~180 N on the floor (pushing/propped), torque saturation 0 joints, joint-limit violations 0. => NOT physically blocked and not torque-limited: the policy sits stably and has no learning signal/path to tuck legs and rise. (Script: `scripts/diag_transition.py`.)
User guidance adopted (18:57): APEX not a training lane (reference only); v7 = controlled pilot, one change at a time (gated per-env assist ON vs OFF first; history/action-scale separately); eval reports assist ON and OFF; matched start checkpoints/budgets; v6 Lane B starting from Lane A is an experiment, not independent histories.

## v7 pilot (launched 19:08)
- Control = A6 (already running). Treatment `A7_gatedassist_fromA1_lr3e4_s1_4096` (`G1-Recovery-A-Assist-v0`, `G1R_ASSIST_TRAIN=1`): single change = HoST-style gated per-env assist (details `lanes/a/NOTES_assist.md`). Eval protocol: assist OFF and ON, force stats reported.

## Results so far (held-out 256 poses, unassisted unless noted; single seed, runs are not bit-repeatable: recovery-only success varied 3-8 of 256 for the same A6 checkpoint across reruns)
| run | legacy success (pelvis>=0.6, tilt<=26deg, 2 s) | STRICT supported stand (+both feet on floor, speed<0.6) |
|---|---|---|
| A v1 (scratch, 1024 envs, 24.6M) | 0 | 0 |
| A5 / B5 (walking-warm, 59M) | 0 / 0 | - |
| A6 (from A1, level 1 start, 59M) | 7 (re-eval 5) | 3 |
| B6 (same init, Lane B reward + global assist) | 3 (re-eval 1) | 1 |
| A7 (A6 + HoST-style gated assist), assist OFF | 7 (re-eval 6) | 0 |
| A7, assist ON (1.0, mean force 7.2 s, peak 186 N) | 17 (re-eval 13) | 0 |
- Chain (A6 -> walking policy, `lanes/a/chain_A6_ho100.json`): 4/256 poses completed fall -> unassisted get-up -> stand 2 s -> walk (10.8 m mean at 0.72 m/s, no falls after hand-over); hand-over after only 10 standing steps: 8 chains, 23 envs fell after hand-over. (Fixed a bug first: rel_standing_envs=1.0 had zeroed the walking command.)
- Stand diagnostic (`lanes/a/standfail_A6.json`): at the first moment the legacy criterion is met, feet are NOT in contact (0-10%), 29-40 cm above the floor, pelvis speed 0.6-1.1 m/s, torso angular speed ~9 rad/s: the legacy criterion is met by ballistic lunges. New primary metric = STRICT supported stand (tracker `succ_s`, eval JSON `strict_supported_stand`).
- v7 pilot conclusion: gated per-env assist did not raise unassisted success at equal budget (both 7/256 legacy, 0-3 strict).
- v8 (launched 19:50): single change = training reward/curriculum use the STRICT criterion (`G1R_STRICT=1`) vs matched control continuing A6 with the legacy criterion (`A8c`), both from A6 model_599, level start 1, 4096 envs x 600 its, lr 3e-4, seed 2.

## v8 results and the exploit finding (19:05-19:15)
- A8c (control: continue A6, legacy reward, seed 2): legacy 256/256, STRICT 215/256, first stand at 0.5 s. A8s (strict-supported reward, seed 2): 0/256 (no gradient; collapsed).
- **A8c is NOT a valid plausible recovery**: `lanes/a/diag_A8c_level4_first1p5s.json`: pelvis peak upward speed mean 3.35 m/s (max 5.6), pelvis 0.24 -> 0.72 m in 1.5 s, joints up to 2.47x their soft velocity limit (right knee), torque at 100% of effort limit, joint-limit hits on 66% of steps. Video frames `media/20261106_190102_A8c_heldout_t0.3s.png` show robots airborne at 0.3 s, standing wide-stance by 1.5 s. It exploits unenforced joint speed (implicit PD, no velocity clamp): a launch/kip-up, not a get-up. Reported as an exploit, not a success.
- New primary metric = PLAUSIBLE recovery: strict supported stand 2 s AND joint-velocity-limit violations <=2% of steps AND peak pelvis upward speed <=2.0 m/s (eval JSON `plausibility`).
- v9 (launched 19:16): single change vs control = joint velocity-limit penalty (`G1R_VELPEN=10`, soft ratio 0.9, Lab `joint_vel_limits`, HoST/APEX-style). Runs: `A9v_velpen_s2`, `A9v_velpen_s3` vs control replicate `A8c_contA6_legacy_s3` (control seed 2 = A8c). All from A6 model_599, level start 1, 4096 envs x 600 its, lr 3e-4.

## 2026-11-06 20:10: restart after server restart; mission update (no push/publish; no GPU coordination gate; cheap helper jobs only)
- Criteria frozen in `CRITERIA.md` (v1) + Amendment 1 (post-hoc free-flight criterion).
- v9 results (A6 init + velocity-limit penalty 10): A9v_s2 0/256 under every criterion (never_rose 125, rose_not_upright 131); A9v_s3 was killed by the server restart at iteration 526 (not evaluated).
- Control replicate A8c seed 3 reproduces the launch exploit (legacy 256/256, strict 240); v1-plausible 52/256, **v1.1-plausible 0/256 for A6, A8c (both seeds), A9v** (flight 3-12% of steps, peak pelvis vz 3-4 m/s). **No plausible recovery exists yet.** The earlier A6 chain (4/256) is retracted: A6's stands are launches.
- v10 launched 20:10 (all from Lane A v1 sit-up policy `A_s1_1024/model_999.pt`, level start 1, seed 2, 4096 envs x 600 its, lr 3e-4): `V10_base_velpen10` (baseline: velocity-limit penalty 10 from the start), `V10_assist_velpen10` (+ gated per-env assist, single change), `V10_velpen3` (penalty 3, single change).

## v10 results (from A1 init, velocity-limit penalty from the start, seed 2, 4096 envs x 600 its; held-out 256; assist OFF unless noted)
| run | legacy | strict | v1 plaus | v1.1 plaus | flight frac | peak pelvis vz (m/s) |
|---|---|---|---|---|---|---|
| V10_base_velpen10 | 82 | 61 | 0 | 0 | 0.16 | 5.0 |
| V10_velpen3 | 19 | 5 | 1 | 0 | 0.13 | 4.1 |
| V10_assist_velpen10, assist OFF | 25 | 16 | 0 | 0 | 0.10 | 3.3 |
| same, assist ON (7.4 s mean force) | 24 | 20 | 0 | 0 | 0.11 | 3.2 |
Conclusion: joint-velocity-limit penalty does not stop the launch (pelvis speed is unconstrained); gated assist does not change it. Launch emerges again from the sit-up policy.
- v11 launched 20:45 (A1 init, velpen 10, seed 2): `V11_launchpen5` (+ penalty on pelvis speed above 1 m/s and on free flight, weight 5), `V11_launchterm2p5` (instead: terminate when pelvis speed > 2.5 m/s, HumanUP terminate_on_velocity idea), `V11_launchpen5_assist` (launchpen + gated assist).

## v11 results (A1 init, seed 2, 4096 envs x 600 its; held-out 256)
| run | legacy | strict | v1.1 PLAUSIBLE | ever supported stand | flight | peak vz | pos-limit steps |
|---|---|---|---|---|---|---|---|
| launchpen5 | 0 | 0 | 0 | 0 | 0.007 | 0.57 | 0.29 |
| launchterm 2.5 | 0 | 0 | 0 | 0 | 0.017 | 0.69 | 0.10 |
| launchpen5 + gated assist, assist OFF | 2 | 2 | **2** | 15 | 0.011 | 0.74 | 0.53 |
| same, assist ON (mean 7.4 s force) | 9 | 9 | 9 | 42 | 0.016 | 0.84 | 0.53 |
First plausible (no-launch) unassisted recoveries: 2/256 (assist-on 9/256). No-launch regulariser removed the exploit (flight <2%, peak pelvis vz <1 m/s); assist now helps (it only did nothing while the launch was available). Caveat: joint-POSITION-limit steps 53% for the assisted policy (not part of criterion v1; mission asks for valid limits -> consider Amendment 2).
- v12 launched 21:25: `V12_cont_assist_1200` (continue V11_assist +1200 its), `V12_cont_noassist_1200` (control: continue V11_launchpen5 +1200), `V12_rep_s3_assist_600` (seed 3 replicate from A1, 600 its). All velpen 10 + launchpen 5; assist per-env level restarts at 1.0 on continuation (critic/optimizer fresh).

## Latest media
- `media/20261106_205009_V11_unassisted_t6s.png`, `media/20261106_205009_V11_unassisted_16poses.gif`: V11 launchpen5+assist policy, assist OFF, 16 held-out poses at 6 s: robots push up into crouched/hand-and-foot postures with no airborne phase (0/16 reach a supported 2 s stand in this clip; 2/256 overall).

## Next
0. Lane A v2: reward/curriculum fix for the sit-down local optimum (stronger height/stand shaping, lighter action penalty, softer promotion rule); keep Lane B as the assist-force variant.
1. Shared base + smoke test (peak memory at startup/train/eval per lane).
2. Launch Lane A and Lane B implementers; concurrent smoke, then scale envs.
3. Evaluate on held-out poses; chain with walking policy; media.

## Blockers
None.

## Runs / commands
- **v5 results (4096 envs, 600 its = 59 M transitions, warm walking actor, lr 3e-4)**: A5 0/256 (never_rose 251; train 1319 s, 44.7k trans/s; mean level 0.56), B5 0/256 (never_rose 250; 1275 s, 46.3k trans/s; mean level 0.91; assist 0.9). Warm-starting from the walking policy erased the lying->sit-up behaviour that from-scratch A v1 had found; still no sit->stand.
- **v6 launched 18:45**: `A6_fromA1_lr3e4_s1_4096`, `B6_fromA1_lr3e4_s1_4096`: both warm-started from the Lane A v1 sit-up policy (`lanes/a/runs/A_s1_1024/model_999.pt`), curriculum START LEVEL 1 (`G1R_START_LEVEL=1`; sit->stand first), 4096 envs x 600 its, lr 3e-4, seed 1. Note: B's init actor therefore comes from an A-lane run (documented asymmetry).
- v5 mid-run (it ~300): A5 mean level flat ~0.45-0.55, stand reward ~1.4-1.9 (level-0 starts succeed, level-1 sit starts do not); B5 mean level 0.5->0.9 but stand reward dropping 0.7->0.02, assist stuck ~0.9. Sit->stand not discovered yet by either lane.
- v4 (A4/B4) aborted after ~100 its: mean curriculum level oscillated 0<->1 because all envs reset in lockstep (trainer used init_at_random_ep_len=False; the walking task uses True). Fixed (random episode phases). Logs kept in lanes/*/runs/_aborted/.
- **v5 launched 18:05**: `A5_warm_lr3e4_s1_4096`, `B5_warm_lr3e4_s1_4096` = v4 settings + random episode phases. 4096 envs x 600 iters (59 M transitions), seed 1, warm actor, fixed lr 3e-4. A alone ran at ~85k trans/s at 4096 envs.
- **Diagnostic (17:55)**: the walking policy (model_1499) applied to the FULL-collision G1 at level 0 (standing start): 16/16 robots stand the full 10 s, so the walking policy balances fine on this model (`lanes/a/logs/rec_walk_L0.log`). In v3 (A3/B3) the warm-started actor LOST that stand during PPO (stand reward 0.59 at ~it20 -> 0.29; level back to 0) = fine-tuning drift (fresh critic + adaptive lr 1e-3). A3/B3 stopped at ~iteration 200/20 (my own PIDs only).
- **v4 launched 17:58 (user asked for more envs; GPU mem ~17 GB of 98 GB)**: `A4_warm_lr3e4_s1_4096`, `B4_warm_lr3e4_s1_4096`: 4096 envs x 600 iters (59 M transitions), seed 1, actor warm start + fixed lr 3e-4 (`--lr 3e-4`), same bank curriculum.
- **v3 launched 17:45**: `A3_warm_s1_2048` and `B3_warm_s1_2048`: 2048 envs x 1000 iters (49.2 M transitions), seed 1, warm-started actor from the walking policy, new bank-based curriculum (bank_v1.npz: 10,025 states harvested from A v1 rollouts, 8.9k sitting) and Lane B assist decay on 'reached standing' EMA > 0.5. Level-1/2 start poses rendered: `media/20261106_173906_level{1,2}_start_poses.png`.
- **v2 results (both warm-started, 24.6 M transitions each, seed 1)**: A2 0/256 (never_rose 83, rose_not_upright 173; train 1108 s, 22.2k trans/s, final mean level 1.11), B2 0/256 (never_rose 252, 81% joint-limit-violation steps; train 1167 s, 21.1k trans/s, assist stuck at 0.92 because success EMA never exceeded 0.30). Both stalled at curriculum level ~1.1 (poses = standing robot tilted 15-70 deg in mid-air = fall-arrest, not get-up) and the standing reward decayed after the first ~100 iterations.
- **Diagnosis -> v3 plan**: (1) replace tilt-in-air levels by L0 stand, L1 sit/crouch (state bank), L2 mid-rise (state bank), L3/L4 lying; bank harvested by running A v1 (which DOES rise lying->sitting) on training-distribution lying poses (not the held-out set; same bank for both lanes, documented); (2) Lane B assist decays on 'reached standing at least once' EMA > 0.5 instead of full success; (3) scale to 2048 envs x 1500 iters (73.7 M transitions) once overlap is verified stable; both warm-started from the walking actor.
- **A_s1_1024 (Lane A v1, from scratch) DONE**: 24.576 M transitions, 827 s wall, 29.7k trans/s (overlapped with Lane B + Jenga). Held-out eval (256 poses, it999): **0/256 success**; failure causes never_rose 73, rose_not_upright 171 (sits up, legs out, never stands), stood_then_fell 12; joint-limit violation steps 17%. File `lanes/a/eval_A_s1_1024_it999.json`. Media: `media/20261106_171247_A_v1_failure.gif`, `..._failure_t{1,4,9}s.png` (policy sits up from lying but cannot stand).
- **B_s1_1024 (Lane B v1, from scratch) DONE**: 24.576 M transitions, 1055 s, 23.3k trans/s (overlapped with A and later A2/B2). Held-out eval: **0/256**; never_rose 252 (assist still at scale 1.0 at the end, so unassisted eval cannot rise), joint-limit violation steps 39%. `lanes/b/eval_B_s1_1024_it999.json`.
- **A2_warm_s1_1024 (Lane A v2)**: same as v1 plus `--init_actor <walking model_1499.pt>` (actor-only warm start; critic/optimizer fresh), started 17:12. - **B2_warm_s1_1024 (Lane B v2)**: same warm start, started 17:13. Both v2 runs at iteration ~485 mean curriculum level ~1.1; B assist scale 0.92 (decaying).
- `B_s1_1024` (Lane B, HoST-style assist+staged rewards): same command with task G1-Recovery-B-v0, started 17:07 MST, runs CONCURRENTLY with the tail of A_s1_1024 (overlap test).
- Lane A v1 observation (iter 400): policy sits upright (torso-up 0.89, pelvis ~0.2 m), standing bonus ~0, per-env pose curriculum stuck at level 0 -> reward/curriculum too hard to discover standing balance; plan: Lane A v2 (see Next).
- `A_s1_1024`: `scripts/gpu_run.py --name A_s1_1024 --est_gb 10 --log lanes/a/logs/train_A_s1_1024.log -- scripts/run.sh train_recovery.py --task G1-Recovery-A-v0 --num_envs 1024 --max_iterations 1000 --seed 1 --out_root lanes/a/runs --run_name A_s1_1024 physics=newton_mjwarp` (started 16:55 MST)
- Eval: `scripts/run.sh eval_recovery.py --task G1-Recovery-A-v0 --checkpoint <model.pt> --num_envs 256 --out <json> physics=newton_mjwarp`

## Latest media
- `media/20261106_175750_A5_it150_level1_t9s.png`: A5 at iteration 150, level-1 (sitting-bank) starts after 10 s: robots recline/lie back, not standing yet (policy early in training).
- `media/20261106_171247_A_v1_failure_t9s.png` (A v1 final: 16 held-out poses, robots sitting up, none standing)
- `media/20261106_165529_heldout_poses_t0.3s.png`
- Lane B implementation running (status file lanes\/b\/STATUS_lane_b.md); literature review finished (no licensed G1 get-up reference; Lane B = HoST-style assist-force + staged-height rewards).
