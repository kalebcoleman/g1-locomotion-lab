# M4: push and friction robustness of the frozen walking policy (evaluation only)

No training, no policy change. `checkpoints/walking_newton_it1499.pt` (sha256 `daa544a8d3c0...`) in its native env `Isaac-Velocity-Flat-G1` (minimal-collision G1, same 123-d observation and 37-d action), seeds 1-3, forward command 0.8 m/s.

## Protocol
- Robots walk for 3 s, then receive ONE horizontal velocity impulse added to the base velocity (same mechanism as M2): magnitude 0 to 4 m/s (15 values), 8 directions 45 deg apart, measured relative to the robot's heading at that moment (0 = forward, 90 = left, 180 = backward, 270 = right), 4 robots per (magnitude, direction) cell per seed. A robot **survives** if it does not fall in the next 5 s. A fall is the environment's torso-contact termination, or pelvis < 0.45 m or torso up < 0.5 for 10 consecutive steps.
- Robots that fall **before** the push never felt it: they are excluded from the push curves and reported as "walked the first 3 s". They are 19 of 1440 (1.3%) at the default friction.
- Friction: the floor's static = dynamic friction and the robots' body material are set to the same value, so the number is the effective foot-floor coefficient. (Training: floor 1.0, robot 0.8 / 0.6.) A first sweep that lowered only the floor had no effect, because contacts use the larger of the two friction values; it is discarded and kept in `invalid_first_friction_run/`.
- M50 = push magnitude where 50% of the robots standing at the push fall: interpolated between the measured shares, and from a maximum-likelihood logistic fit with a 95% percentile bootstrap interval (400 resamples of the binomial counts). The interval covers sampling noise only, not run-to-run simulation nondeterminism.

## Push curve at the training friction (1.0), pooled over directions and seeds
| push (m/s) | survived (standing at the push) | survival [95% Wilson CI] |
|---|---|---|
| 0 | 96/96 | 100% [96, 100] |
| 0.25 | 95/95 | 100% [96, 100] |
| 0.5 | 95/95 | 100% [96, 100] |
| 0.75 | 96/96 | 100% [96, 100] |
| 1 | 92/95 | 97% [91, 99] |
| 1.25 | 86/94 | 91% [84, 96] |
| 1.5 | 68/92 | 74% [64, 82] |
| 1.75 | 60/96 | 62% [53, 72] |
| 2 | 49/95 | 52% [42, 61] |
| 2.25 | 28/95 | 29% [21, 39] |
| 2.5 | 29/94 | 31% [22, 41] |
| 2.75 | 22/95 | 23% [16, 33] |
| 3 | 20/94 | 21% [14, 31] |
| 3.5 | 11/94 | 12% [7, 20] |
| 4 | 6/95 | 6% [3, 13] |

**M50 at friction 1.0: 2.02 m/s interpolated, 2.16 m/s logistic [95% CI 2.08, 2.24].** Survival is 97% at 1 m/s, 74% at 1.5 m/s, 52% at 2 m/s and 6% at 4 m/s. Training used random pushes of up to 0.5 m/s in x and y every 10-15 s (the task's own `push_robot` event; `docs/WALKING.md`), so the largest training push is about 0.71 m/s combined and 2.16 m/s is about three times that.

## By push direction (friction 1.0)
| push direction (relative to heading) | M50 interpolated | M50 logistic | falls at 2.0 m/s |
|---|---|---|---|
| forward (0 deg) | 2.58 m/s | 2.85 m/s | 2/11 |
| forward-left (45) | 2.85 m/s | 2.77 m/s | 1/12 |
| left (90) | 1.43 m/s | 1.48 m/s | 11/12 |
| rear-left (135) | 1.35 m/s | 1.40 m/s | 11/12 |
| backward (180) | 2.05 m/s | 1.94 m/s | 5/12 |
| rear-right (225) | 2.00 m/s | 2.05 m/s | 6/12 |
| right (270) | 1.75 m/s | 1.77 m/s | 9/12 |
| forward-right (315) | 3.62 m/s | 3.44 m/s | 1/12 |
Left and rear-left pushes are the weakest (M50 about 1.4 m/s, 11 of 12 fall at 2.0 m/s); forward and forward-left pushes are the strongest (M50 about 2.6 to 3.6 m/s). The cells have 11 to 12 robots at 2.0 m/s, so single directions carry wide uncertainty; the left/right asymmetry (1.4 vs 1.75 m/s) is as measured, not explained.

## Friction
| floor and body friction | walked the first 3 s without falling | standing at the push, per magnitude | M50, interpolated | M50, logistic [95% bootstrap CI] |
|---|---|---|---|---|
| 1.0 | 1421/1440 (99%) [98, 99] | 92-96 | 2.02 m/s | 2.16 [2.08, 2.24] m/s |
| 0.8 | 1421/1440 (99%) [98, 99] | 92-96 | 1.87 m/s | 2.10 [2.04, 2.17] m/s |
| 0.6 | 1422/1440 (99%) [98, 99] | 92-96 | 1.87 m/s | 1.97 [1.90, 2.03] m/s |
| 0.5 | 1420/1440 (99%) [98, 99] | 92-96 | 1.67 m/s | 1.82 [1.76, 1.88] m/s |
| 0.4 | 1416/1440 (98%) [98, 99] | 92-96 | 1.66 m/s | 1.68 [1.63, 1.74] m/s |
| 0.3 | 1330/1440 (92%) [91, 94] | 85-92 | 1.41 m/s | 1.45 [1.39, 1.50] m/s |
| 0.2 | 806/1440 (56%) [53, 59] | 47-60 | 0.90 m/s | 0.96 [0.90, 1.01] m/s |
| 0.1 | 51/1440 (4%) [3, 5] | 2-6 | 0.25 m/s | 0.26 [0.26, 0.26] m/s |

- Friction at or above 0.4 changes little: 98-99% walk the first 3 s, M50 falls gradually from 2.16 to 1.68 m/s.
- Below that the policy breaks down quickly: 92% walk the first 3 s at 0.3, 56% at 0.2, 4% at 0.1. With no push at all, no robot that was standing at 3 s falls afterwards (0 of 96 down to 0.4, 0 of 90 at 0.3, 0 of 47 at 0.2).
- At friction 0.2 and 0.1 only 47-60 and 2-6 robots per magnitude are still standing at the push, so the M50 values there (0.96 and 0.26 m/s) rest on a few robots that already coped with the floor and are not reliable. The 0.1 interval is degenerate for that reason.

## Limits
- Three seeds, 4 robots per cell per seed; the confidence intervals above are binomial, not a bound on simulation nondeterminism.
- A single push at one phase of the gait (3.0 s after the start, not synchronised to the steps); direction effects may depend on the phase.
- The effective friction coefficient is a single value for floor and robot; real contact pairs differ. Restitution, mass and motor-strength variation are not swept.
- In the videos a robot that falls is reset by the environment's termination, so later frames show it standing again at its start position; the statistics use each robot's first fall only.
- Video: one 16-robot run (seed 5; magnitudes 0, 1.5, 2.5, 3.5 m/s x 4 directions), uncut. One of the four no-push robots fell during its first 3 s (a start-up fall); the 1.5 m/s robots survived and the 2.5 and 3.5 m/s robots mostly fell.

## Reproduce
```
./scripts/run.sh eval_robustness.py --task Isaac-Velocity-Flat-G1 --walk_checkpoint <walking_newton_it1499.pt> --seed 1 --reps 4 --friction 1.0 --out <out.json> physics=newton_mjwarp
python pilots/m4_robustness_20261108/aggregate_robustness.py          # reads results/m4_mu*_s*.json
# video: --reps 1 --n_dirs 4 --mags 0,1.5,2.5,3.5 --record_dir <dir> --label <name>
```
`ledger.csv`: wall seconds per job (about 20 s each).
