# M2: frozen-policy walk -> induced fall -> get-up -> stand gate -> walk (evaluation only)

No training. One continuous episode per robot with no environment reset; a reset anywhere counts as a failure. DEV-style evaluation: no pose set is used (the fall comes from the dynamics), so DEV, HELDOUT and TEST sets are untouched. Seeds 2-4 for the numbers below; seed 1 was used for calibrating push magnitudes only.

## What runs
- Policies: walking `checkpoints/walking_newton_it1499.pt` (sha256 `daa544a8d3c0...`), recovery `R9hg_it1100` (sha256 `665a89f58b3a...`). Both frozen.
- Env: `G1-Recovery-R9hg-v0` (full-collision G1, self-collision on, Unitree torque limits, Newton/MJWarp, 50 Hz). The walking policy was trained on the minimal-collision G1; here it runs on this env. The no-push control shows it does (below).
- Compatibility: both policies read the same 123-d observation from the same env (base linear velocity 3, angular velocity 3, projected gravity 3, velocity command 3, joint positions 37, joint velocities 37, last action 37) and emit 37-d joint-position targets through the same action term (scale 0.5, default offset, no clip). Neither checkpoint contains an observation normalizer. The recovery policy gets a zero velocity command; the walking policy gets `vx = 0.8 m/s`.
- Timeline: robots start standing; the walking policy walks; at 3.0 s a push is applied; fall detection; recovery policy; stand gate; one-way hand-over; walking for 10 s.

## How the fall is induced
At t = 3.0 s a horizontal velocity impulse is added to the base velocity (same mechanism as Isaac Lab's `push_by_setting_velocity`, with a logged per-env vector): magnitude 2.5, 3.5 or 4.5 m/s (cycled over envs), random direction per env fixed by the seed. Magnitude 0 is the no-push control. Magnitudes were chosen from a 16-env calibration (seed 1): 1.5 m/s did not make robots fall; 3.0 and 4.5 m/s did. No trip method is implemented.

## State machine (`g1_recovery/fall_handover.py`, 15 CPU unit tests)
- Fall: pelvis < 0.45 m or torso up < 0.5 for 10 consecutive steps (0.2 s). Latency is measured from the push to this detection.
- Hysteresis: the fall thresholds (0.45 m, 0.5) are below the stand thresholds (0.60 m, 0.90). Transitions are one-way: recovery never returns to walking and walking never returns to recovery; after hand-over only the debounced fall thresholds count as a re-fall.
- Gate: criteria v3 stand/hold measured over the get-up window (from fall detection): supported 2 s stand (both feet loaded, pelvis >= 0.60 m, up >= 0.90, 3D speed < 0.6 m/s) and, over the window, joint-velocity limit <= 2% of steps, ground flight <= 2%, position limit <= 10%, hard-limit rule, torque <= 3%, peak upward speed <= 2.0 m/s, mean foot distance over the hold <= 0.60 m. 3b self-penetration is not applied (offline screen). Tracker counters are restarted at fall detection (bookkeeping, not an env reset).
- Success: hand-over, then 10 s of walking with no re-fall, >= 3.0 m along the heading at hand-over and >= 0.5 m/s mean speed (the Stage 2 chain rule).
- Time limits: 3 s after the push to fall, 10 s to pass the gate, 10 s of walking. Any unfinished episode is a failure.

## Results (seeds 2-4, 128 envs each: 96 no-push controls and 288 pushed; criteria-v3 gate)
| push (m/s) | episodes | fell | gate passed (hand-over) | re-fell after hand-over | walked 10 s | full success |
|---|---|---|---|---|---|---|
| 0 (control) | 96 | 0 | - | - | 96 of 96 still walking | - |
| 2.5 | 96 | 60 (62%) | 29 | 20 | 9 | 2 |
| 3.5 | 96 | 94 (98%) | 45 | 28 | 17 | 1 |
| 4.5 | 96 | 96 (100%) | 24 | 13 | 11 | 3 |
| all pushed | 288 | 250 | 98 | 61 | 37 | **6 of 250 (2.4%)** |

- Environment resets: 0. Detection latency (push to detection): 0.78 / 0.70 / 0.57 s mean for 2.5 / 3.5 / 4.5 m/s. Get-up time (detection to gate), when it passed: about 7.3 to 8.0 s mean.
- Why most episodes stop at the gate: 152 robots timed out; all 152 had reached a supported 2 s stand (the robots are standing), but failed v3 validity clauses over the window. Clauses failed (a robot can fail several): ground flight 129, hard-limit excursion > 0.10 rad 74, hard-limit steps > 3% 48, position limit 23, launch 4.
- After hand-over: 61 of 98 re-fell, at a mean of 1.66 s after hand-over (51 within 2 s). 37 completed the 10 s window, covering 4.0 m on average (31 reached 3.0 m, 6 reached 5.0 m). The no-push control walks 6.16 m on average over the same window (96 of 96 reach 5.0 m), so the walking policy itself is not the limit. Of the 37 completed, 6 passed the speed rule (25 too slow, 6 too little progress).
- Spread over seeds: the 6 successes are all from seed 3 (seeds 2 and 4: 0). Per seed, induced falls 84 / 83 / 83, hand-overs 37 / 33 / 28.

Exploratory (not criteria v3): gate = supported 2 s stand only, same seeds.
| push (m/s) | fell | hand-over | re-fell | walked 10 s | full success |
|---|---|---|---|---|---|
| 2.5 | 59 | 59 | 40 | 19 | 4 |
| 3.5 | 93 | 93 | 60 | 33 | 13 |
| 4.5 | 96 | 96 | 56 | 40 | 8 |
| all | 248 | 248 | 156 | 92 | **25 of 248 (10.1%)** |
Get-up time to a 2 s stand is about 3.1 to 3.4 s. 156 of 248 re-fell, mean 1.50 s after hand-over (145 within 2 s). The v3 validity clauses cost 150 hand-overs; the hand-over itself remains the main failure either way.

## M1 movement metrics during the get-up (mean over windows; v3 run, n = 250)
Peak foot spread after the pelvis passes 0.40 m 1.33 m (whole window above 0.12 m: 1.40 m, at 0.56 s mean); final spread 0.43 m; share of gated steps wider than 0.45 m 0.42; hip roll / yaw peak 1.44 / 1.37 rad (includes the start of the window); action rate 0.182; joint acceleration RMS / p99 66 / 298 rad/s^2; jerk p99 12,600 rad/s^3. The wide-legged transient measured from lying starts (1.32 m) is also present when getting up from an induced fall.

## Limits
- One recovery policy (R9hg it1100, a single seed) and one walking checkpoint; 3 evaluation seeds, 384 episodes. Counts are small (6 successes) and clustered in one seed, so success rates are indicative only.
- The batch was run twice (the second after fixing a re-fall labelling bug and adding the control reference); the numbers above are the second run. Gate passes differed between runs (98 vs 80 hand-overs under v3), successes were 6 in both: expect roughly +-10% run-to-run variation from simulation nondeterminism.
- Push magnitudes were chosen so that most robots fall; they are not a model of real disturbances. No trip, no friction or terrain variation (M3, M4).
- The v3 gate here is applied to a window that starts at fall detection, not to a 10 s episode from a fixed pose, so its clause fractions are over a shorter, variable window.
- The video is one 16-env run of seed 3 (4 controls, 4 each of 2.5, 3.5, 4.5 m/s), uncut, chosen because the batch had a success among its first 16 envs; its own outcomes (2 successes, 1 re-fall, 8 timed-out recoveries, 1 no-fall, 4 walking controls) differ slightly from the batch because runs are not bit-repeatable.

## Reproduce
```
# from recovery/ ; both checkpoints from checkpoints/
./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint <R9hg_it1100.pt> --walk_checkpoint <walking_newton_it1499.pt> \
    --num_envs 128 --seed 2 --levels 0,2.5,3.5,4.5 --gate v3 --out <out.json> physics=newton_mjwarp      # --gate strict for the exploratory run
./scripts/run.sh eval_fall_handover.py ... --num_envs 16 --seed 3 --record_dir <dir> --label <name> --out <out.json>
./scripts/contact_sheet_seq.sh <mp4> <png> 2.9 3.6 4.4 5.5 7 10 14 22
python pilots/m2_fall_handover_20261108/aggregate.py v3 results/m2_v3_s2.json results/m2_v3_s3.json results/m2_v3_s4.json
```
Files: `results/` (per-seed JSON with per-env records, pooled tables, calibration), `ledger.csv` (wall seconds per job, about 6 min total GPU for the batch, calibration and video), `media/` (contact sheet; the MP4 was delivered separately).
