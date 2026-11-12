# G1 fall-recovery experiments: report (state as of 2026-11-06 ~21:45 MST)

> **Historical snapshot.** This document describes the project as of 2026-11-06. Current results are in [docs/RESULTS.md](../docs/RESULTS.md).

Simulation only (Isaac Lab 3.0.0, Newton/MJWarp, Unitree G1). Goal: **unassisted fall -> get up -> stable stand -> walk**, then walk -> disturbance -> fall -> recover -> walk/turn.
**Status in one paragraph:** a *plausible* (no launch, feet-supported, unassisted) get-up has been demonstrated on only **2 of 256 held-out poses** so far (9 of 256 with a gated assist applied), at one seed. No stand->walk chain and no continuous disturbance sequence have been demonstrated under the plausibility criteria. Two earlier "successes" (A6's 4/256 chain, A8c's 256/256) were **rejected**: A8c launches itself upward (peak pelvis speed 3-6 m/s, joint speeds up to 2.5x the soft limit) and A6's stands were also launches. Runs v12 (longer training + seed replicate) were still training when this report was written.

## 1. Setup
| | |
|---|---|
| Software | Isaac Lab 3.0.0 (commit e3f7d9f), Isaac Sim 6.2.0, Newton 1.6.1, mujoco_warp 3.12.0, MuJoCo 3.12.0, Warp 1.17.0, rsl-rl-lib 5.5.1, PyTorch 2.11.0+cu130, Python 3.12, Ubuntu 22.04 |
| Hardware | Intel Xeon w5-3435X, 125 GB RAM, one NVIDIA RTX PRO 6000 Blackwell Max-Q (98 GB, driver 595.99.02), **shared with other jobs** |
| Robot | `G1_CFG` (full collision g1.usd, 37 joints incl. hands). The walking task's `G1_MINIMAL_CFG` has only feet+torso collision boxes (3 shapes), unusable for lying poses |
| Physics/control | `physics=newton_mjwarp`, dt 0.005, decimation 4 (50 Hz), joint-position actions (scale 0.5 about default), contacts via Newton collision pipeline, `njmax 400 / nconmax 80` |
| Observation | the walking policy's 123-d vector (zero velocity command) so a recovery and the walking policy can be chained |
| Policy/PPO | Lab's G1 flat PPO (MLP 256-128-128), `lr 3e-4` fixed in the later runs, 4096 envs x 24 steps per iteration |
| Episode | 10 s, no pushes in recovery training; start poses from a per-env pose curriculum (levels: stand, sit/crouch bank, mid-rise bank, lying) |
| Held-out set | `heldout_poses.npz`: 256 lying poses, 4 fall directions (supine, prone, two sides) x 64, limbs +-0.4 rad, seed 20261106, never used in training |
Exact commands: `scripts/run.sh train_recovery.py --task G1-Recovery-A-v0 --num_envs 4096 --max_iterations 600 --seed S --lr 3e-4 --init_actor <ckpt> --out_root <dir> --run_name <name> physics=newton_mjwarp` with env vars `G1R_START_LEVEL=1`, `G1R_VELPEN=10`, `G1R_LAUNCHPEN=5`, `G1R_ASSIST_TRAIN=1` (task `G1-Recovery-A-Assist-v0`) as listed per run in `STATUS.md`. Evaluation: `scripts/eval_recovery.py` (add `G1R_EVAL_ASSIST=0|1.0` for the assist task), chain: `scripts/eval_chain.py`, diagnostics: `diag_transition.py`, `diag_standfail.py`. All launched through `scripts/gpu_run.py` (memory/throughput logging).

## 2. Success criteria (frozen before evaluating; `CRITERIA.md`)
Plausible unassisted recovery = supported stand (pelvis >= 0.60 m, tilt <= 26 deg, **both feet** > 10 N and < 0.15 m, pelvis speed < 0.6 m/s) held 2 s, with <= 2% joint-velocity-limit-violating steps, peak pelvis upward speed <= 2.0 m/s, no assist. **Amendment 1 (post-hoc, labelled):** plus free flight (no body touching anything) on <= 2% of steps ("v1.1"). Legacy (pelvis+upright only) and strict (supported) counts are always shown next to it.

## 3. Results (held-out 256 poses, assist OFF unless noted; one seed per row; evaluation is not bit-repeatable: A6 scored 3-8/256 across reruns)
| run (init, budget) | legacy | strict supported | **v1.1 plausible** | notes |
|---|---|---|---|---|
| Lane A v1: scratch, 1024 envs, 24.6 M | 0 | 0 | 0 | learns lying -> long sitting (171/256 rose partway) |
| Lane B v1 (HoST-style global assist) scratch | 0 | 0 | 0 | assist never decayed |
| v2 / v5 warm-started from walking policy (24.6 M / 59 M) | 0 | 0 | 0 | warm start erased the sit-up skill |
| A6: from A1, level-1 start, 59 M | 3-8 | 3 | 0 | stands are launches (flight 12%, peak vz 4.1 m/s) |
| B6 (Lane B reward + global assist) | 1-3 | 1 | 0 | |
| A7: A6 + gated per-env assist (treatment) | 6-7 | 0 | 0 | with assist on: 13-17 legacy, 0 strict |
| **A8c: continue A6 (control) seed 2 / seed 3** | **256 / 256** | 212-218 / 239-240 | **0 / 0** | **launch exploit, reproduced on 2 seeds; rejected** |
| A8s: strict-support reward | 0 | 0 | 0 | no gradient |
| A9v: A6 + joint-velocity penalty | 0 | 0 | 0 | removes launch, nothing replaces it |
| V10 base / pen3 / assist (A1 init, vel. penalty) | 82 / 19 / 25 | 61 / 5 / 16 | 0 / 0 / 0 | launch returns (peak vz 3-5 m/s) |
| V11 launch-penalty (pelvis speed + free flight) | 0 | 0 | 0 | exploit gone (flight 0.7%, peak vz 0.57) |
| V11 launch-termination (2.5 m/s) | 0 | 0 | 0 | |
| **V11 launch-penalty + gated assist, assist OFF** | 2 | 2 | **2** | ever reached supported stand: 15/256 |
| same, assist ON (mean 7.4 s of force, peak 186 N) | 9 | 9 | 9 | ever supported: 42/256 |
**Current best checkpoint:** `checkpoints/V11_launchpen5_assist_it599.pt` (task `G1-Recovery-A-v0`; assist is training-only). Caveat: it violates joint *position* limits on ~53% of steps (not part of the frozen v1 criteria; the mission asks for valid limits, so this is a validity concern to address, e.g. an Amendment 2).

## 4. What we learned
1. **Metric design decided the headline.** Pelvis-height+upright alone is satisfied by a ballistic lunge; the first diagnostic (`standfail`) showed feet 30-40 cm off the floor, pelvis 0.6-1.1 m/s and ~9 rad/s torso spin at the "stand" moment. Requiring both feet on the floor, then a free-flight check, removed all fake successes.
2. **PPO finds the cheapest exploit.** With unenforced joint speeds (implicit PD, no velocity clamp) it launched itself within 0.5 s; joint-velocity penalties alone did not stop it because pelvis motion was unconstrained. A direct pelvis-speed + free-flight penalty did (HumanUP also constrains velocity directly).
3. **After the exploit was closed, the gated per-env assist (HoST idea) started to matter.** Before that it made no difference (A7 vs A6, V10 assist vs base), consistent with the policy having an easier shortcut.
4. **Not physically blocked.** At the stuck sit state (Lane A v1) no joint hit torque or position limits and the feet were in contact, i.e. a missing learning signal/exploration, not infeasibility.
5. **Warm starting from the walking policy hurt** (kept standing, lost lying -> sitting). Starting from the sit-up policy helped.
6. **Reference data:** no licensed G1 get-up motion; LAFAN1 retargets (CC BY-NC-ND) deliberately not used. HoST/HumanUP code/checkpoints are Isaac Gym on a 23-DoF G1 and were only read; APEX-humanoid's `standup.pt` (PhysX, 29-DoF, prone start, 558-d history observation, 23 outputs) assessed statically and judged a bigger port than a quick baseline (`research/`).

## 5. Compute profiling (shared GPU, not a benchmark)
~2.6 GB VRAM per 1024-env training process; two or three concurrent 4096-env jobs plus other users' jobs stayed below ~20 GB of 98 GB; a 16-env video render ~4.1 GB; 256-env evaluation ~0.9 GB. The GPU is compute-saturated (88-99%), so more environments or jobs do not add throughput: one job alone at 4096 envs ran ~85 k transitions/s, two to three overlapped jobs 41-80 k each. No out-of-memory error occurred. Run-level wall times and throughput are in `results/train_summaries/`.

## 6. Media (`media/`)
`..heldout_poses_t0.3s.png` (held-out starts), `..A_v1_failure_*` (sits up, no stand), `..level1_start_poses.png` (sit-bank starts), `..A8c_*` (**the launch exploit**: airborne at 0.3 s), `..V11_unassisted_*` (current plausible partial behaviour: pushes up into crouched hand-and-foot postures, no flight). No uncut success video exists yet; none will be presented until measurements support it.

## 7. Limitations
Single seed per condition; evaluation noise large; the 2/256 is far from robust. Chain and disturbance stages untested under the plausibility criteria. Weak joint-position-limit validity. The pose bank came from a Lane A policy (used for both lanes). Lane B initial weights in some runs came from Lane A.

## 8. Next steps (also in `STATUS.md`)
1. Finish v12 (longer continuation of the assisted run and its no-assist control; seed-3 replicate) and evaluate with >= 3 evaluations per checkpoint.
2. Add the joint-position-limit criterion/penalty; re-test.
3. If a plausible rise is repeatable: tighten the success hold, then chain with the walking policy (hand-over only on a plausible supported stand; check the rel_standing_envs/command bug is fixed, as it is in this repo).
4. Then the stretch sequence (walk -> push -> fall -> recover -> walk/turn) with fixed held-out disturbances, >= 3 seeds, uncut video.
5. Untried: observation history, action-scale curriculum, unactuated settle (HoST), per-family pose training, self-tracking (HumanUP stage II) from plausible rollouts.
