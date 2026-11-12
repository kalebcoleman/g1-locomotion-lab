# M3: terrain baseline of the frozen walking policy (evaluation only)

No training, no policy change. The walking policy `checkpoints/walking_newton_it1499.pt` (sha256 `daa544a8d3c0...`, trained on flat ground, 123-d observation, 37-d action) is evaluated blind on graded terrain, seeds 1-3, 8 robots per terrain cell per seed (24 per cell), 10 s, forward command 0.8 m/s, pushes off.

## Setup
- Env: `Isaac-Velocity-Rough-G1` (same minimal-collision G1, same 37-d joint-position action) with the height scan removed, so the policy receives the same 123-d observation it was trained on. The heading command is off. The standard reset event gives random yaw and a +-0.5 m offset on the centre platform of each sub-terrain.
- Robots start on the centre platform of a 20 x 20 m sub-terrain and walk outward. Difficulty `d = (row + U(0,1)) / 8`, so each level covers a band, not a point. The values below are band centres.
- Types: pyramid / inverted-pyramid slope (ratio `0.5 d`, up to 26.6 deg), pyramid / inverted-pyramid stairs (step height `2 + 12 d` cm, step width 0.4 m), random uniform ground (one column per noise amplitude, 0.2 m grid), flat reference. "Down" means the surface falls away from the start (pyramid), "up" means it rises (inverted pyramid).
- A robot **survives** if the environment never terminates it (torso contact) within 10 s. Distance = floor-plane displacement along the initial heading (distance reached before the fall for robots that fall); speed = distance / 10 s. The distance and speed columns are for survivors.

## Results (survived 10 s of 24 per cell, pooled over seeds 1-3)
Flat reference: 24/24, 6.31 m (0.63 m/s).

| level (slope / step height) | slope down | slope up | steps down | steps up |
|---|---|---|---|---|
| 0: 2 deg / 2.8 cm | 23/24 (96%) | 24/24 (100%) | 23/24 (96%) | 20/24 (83%) |
| 1: 5 deg / 4.3 cm | 24/24 (100%) | 24/24 (100%) | 24/24 (100%) | 0/24 (0%) |
| 2: 9 deg / 5.8 cm | 24/24 (100%) | 22/24 (92%) | 24/24 (100%) | 0/24 (0%) |
| 3: 12 deg / 7.3 cm | 24/24 (100%) | 11/24 (46%) | 19/24 (79%) | 0/24 (0%) |
| 4: 16 deg / 8.8 cm | 23/24 (96%) | 7/24 (29%) | 19/24 (79%) | 0/24 (0%) |
| 5: 19 deg / 10.3 cm | 20/24 (83%) | 1/24 (4%) | 16/24 (67%) | 0/24 (0%) |
| 6: 22 deg / 11.8 cm | 17/24 (71%) | 1/24 (4%) | 9/24 (38%) | 0/24 (0%) |
| 7: 25 deg / 13.3 cm | 8/24 (33%) | 1/24 (4%) | 6/24 (25%) | 0/24 (0%) |

| random ground, noise amplitude | survived 10 s [95% Wilson CI] | distance of survivors | speed |
|---|---|---|---|
| 1 cm | 24/24 (100%) [86, 100] | 6.55 m | 0.65 m/s |
| 2 cm | 23/24 (96%) [80, 99] | 6.49 m | 0.65 m/s |
| 3 cm | 23/24 (96%) [80, 99] | 6.03 m | 0.60 m/s |
| 4 cm | 24/24 (100%) [86, 100] | 6.46 m | 0.65 m/s |
| 6 cm | 24/24 (100%) [86, 100] | 5.46 m | 0.55 m/s |
| 8 cm | 22/24 (92%) [74, 98] | 5.00 m | 0.50 m/s |
| 10 cm | 12/24 (50%) [31, 69] | 2.52 m | 0.25 m/s |
| 14 cm | 1/24 (4%) [1, 20] | 5.90 m | 0.59 m/s |

Per-seed rates, Wilson intervals, fall times and distances for every cell: `results/m3_tables.txt` and `results/pooled.json`; per-robot records in `results/m3_s*.json`.

## Hardest level with at least 80% survival
The rule requires every easier level to pass as well, so one lucky hard level above a failing one does not count.
| terrain | hardest level | descriptor | survivors' distance / speed there |
|---|---|---|---|
| slope down | 5 | 19 deg | 9.03 m / 0.90 m/s |
| slope up | 2 | 9 deg | 2.56 m / 0.26 m/s |
| steps down | 2 | 5.8 cm | 6.95 m / 0.70 m/s |
| steps up | 0 | 2.8 cm | 5.50 m / 0.55 m/s |
| random ground | 5 | 8 cm noise | 5.00 m / 0.50 m/s |
| the four graded types pooled by level | 0 | 2 deg / 2.8 cm | (level 1 pools to 75%, because steps up fail from 4.3 cm) |

Survival alone overstates the up-slope result: at 9 deg, 92% survive but cover only 2.6 m in 10 s, i.e. most of them barely progress. Requiring at least 3 m of progress as well, the hardest up-slope level is 1 (5 deg, 5.2 m); the other rows are unchanged.

## What this says
- Down slopes and down steps are handled up to about 19 deg and 6 cm. Random ground is handled up to about 8 cm of noise.
- Up slopes are the weak case beyond about 9 deg, and **up steps fail from 4.3 cm**, with all 168 robots at levels 1-7 falling. This is blind walking by a policy trained on flat ground only: it has no height information.
- Random-ground survival does not fall gradually: 92% at 8 cm, 50% at 10 cm, 4% at 14 cm.

## Limits
- Three seeds, 24 robots per cell: a cell's 95% interval is wide (for example 24/24 is [86, 100]). Terrain variation within a level (the random band and the random noise) is part of the spread.
- Slope and stair surfaces are meshes of 20 m patches; the stairs keep rising for the whole walk (more than 2 m of rise by the end of the walk at the top levels), so a failure may come from any step during 10 s, not the first.
- One walking checkpoint; the policy was never trained on terrain, so this is a baseline for a future terrain fine-tune, not a verdict on the method.
- In the video a robot that falls is reset by the environment's termination, so later frames show it back at its start; the statistics use each robot's first termination only.
- `media/slope_down_19deg.gif` and `media/steps_up_4cm.gif` (with `media/portfolio_slope19_steps43_sheet.png`): two illustrative 8-robot runs recorded for the README (seed 5, `--only slope_down:5` and `--only steps_up:1`, `--reps 8`, `--cam 0,-14,11`): 8 of 8 survived the 19 degree down-slope (10 s); 1 of 8 survived the 4.3 cm step-ups. The pooled evaluation above is the result; these are single runs. Fallen robots are reset by the environment, so later frames show them back at the start.
- The video is one 16-robot run (seed 5, 4 robots on each of slope down 12 / 16 deg and slope up 12 / 16 deg), uncut. The terrain renders pale, so the robots are small; the sheet is evidence of behaviour only (2 of 4 survive on each up-slope, 4 of 4 on each down-slope in that run).

## Reproduce
```
./scripts/run.sh eval_terrain.py --task Isaac-Velocity-Rough-G1 --walk_checkpoint <walking_newton_it1499.pt> --seed 1 --reps 8 --out <out.json> physics=newton_mjwarp
python pilots/m3_terrain_baseline_20261108/aggregate_terrain.py results/m3_s1.json results/m3_s2.json results/m3_s3.json
# video: add  --only slope_down:3,slope_down:4,slope_up:3,slope_up:4 --reps 4 --cam 0,-30,30 --record_dir <dir> --label <name>
```
`ledger.csv`: wall seconds per job (about 1 min each; 3 min for the table).
