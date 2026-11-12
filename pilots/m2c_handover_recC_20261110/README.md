# M2c: the M2b hand-over evaluation rerun with the recC get-up baseline (evaluation only)

Question: M2b measured walk -> induced fall -> get-up -> stand gate -> walk with the **previous** get-up checkpoint (`R9hg_it1100`). The get-up baseline has since been replaced by `getup_baseline_recC_it1425.pt` (held-out criteria-v3 get-up 68.2% vs 50.5%). Does the better get-up translate into better **end-to-end** hand-over success?

No training, no weight changes: the walking policy (`walking_newton_it1499.pt`) and the recovery policy (`recovery/checkpoints/getup_baseline_recC_it1425.pt`, sha256 `a518419e...`) are used as-is. Task `G1-Recovery-R9hg-v0`, Newton/MJWarp, 128 envs per seed, pushes 0 / 2.5 / 3.5 / 4.5 m/s at 3.0 s, DEV-style (no pose sets, TEST untouched). **The protocol, variants, seeds and runner are identical to M2b; only `--checkpoint` differs.**

## Setup
- Variants (unchanged from M2b): `base` = no flags (v3 trigger, M2 behaviour); `blendupper50` = `--settle 50 --blend 50 --blend_joints upper` (v3 trigger); `ready_upper50` = `--trigger ready --settle 50 --blend 50 --blend_joints upper` (readiness trigger, **not** the official v3 stand result).
- Runs: 3 independent repeats (r1, r2, r3) of seeds 2, 3, 4 plus one pass on fresh seeds 5, 6, 7 = 12 runs per variant, 36 runs total, all rc=0, all reported below. One GPU job at a time, about 30 s per job (`ledger.csv`).
- Files: `results/m2c_<variant>_<run>_s<seed>.json`; aggregate in `results/aggregate_m2c.txt` (`python3 aggregate_m2c.py --detail --dist base blendupper50 ready_upper50`); `logs/` is gitignored. The M2b `aggregate_variants.py` reads `results/m2b_<tag>_s<seed>.json` from its own pilot directory and has no repeat dimension, so it does not read these files; `aggregate_m2c.py` is an adapted copy with the same columns plus the Wilson interval and the per-run blocks.

## Result, pooled over the three repeats of seeds 2-4 (9 runs, 1152 envs per variant)
Full success = hand-over, then 10 s of walking with no re-fall, >= 3 m along the heading and >= 0.5 m/s (the M2 rule). `walked 10 s` = the walking window completed without a re-fall. Wilson 95% intervals are on the success rate of induced falls (helper from `recovery/g1_recovery/robustness_stats.py`). M2b rows are the M2b README numbers (pooled row recomputed with the same definition for the columns the M2b README gives only per run).

| variant | checkpoint | induced falls | hand-overs | re-fell | mean t-to-re-fall | walked 10 s | full success | rate (95% Wilson CI) |
|---|---|---|---|---|---|---|---|---|
| base | R9hg (M2b) | 746 | 248 | 132 | 1.61 s | 116 | 19 | 2.5% (1.6-3.9%) |
| base | **recC** | 749 | 420 | 412 | 0.96 s | 8 | 3 | 0.4% (0.1-1.2%) |
| blendupper50 | R9hg (M2b) | 749 | 215 | 0 | - | 215 | 215 | 28.7% (25.6-32.0%) |
| blendupper50 | **recC** | 750 | 387 | 0 | - | 387 | 387 | 51.6% (48.0-55.2%) |
| ready_upper50 | R9hg (M2b) | 754 | 753 | 0 | - | 753 | 753 | 99.9% (99.3-100.0%) |
| ready_upper50 | **recC** | 748 | 748 | 0 | - | 748 | 748 | 100.0% (99.5-100.0%) |

## Every run, both checkpoints (full successes and per-seed full successes)
| variant | run | seeds | R9hg falls | R9hg success | R9hg rate | R9hg per seed | recC falls | recC success | recC rate | recC per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1 | 2-4 | 247 | 2 | 0.8% | [1, 1, 0] | 249 | 0 | 0.0% | [0, 0, 0] |
| base | 2 | 2-4 | 251 | 6 | 2.4% | [2, 0, 4] | 250 | 1 | 0.4% | [1, 0, 0] |
| base | 3 | 2-4 | 248 | 11 | 4.4% | [5, 4, 2] | 250 | 2 | 0.8% | [1, 0, 1] |
| base | pooled | 2-4 | 746 | 19 | 2.5% | | 749 | 3 | 0.4% | |
| base | fresh | 5-7 | 243 | 2 | 0.8% | [2, 0, 0] | 242 | 0 | 0.0% | [0, 0, 0] |
| blendupper50 | 1 | 2-4 | 248 | 73 | 29.4% | [22, 31, 20] | 248 | 112 | 45.2% | [32, 37, 43] |
| blendupper50 | 2 | 2-4 | 252 | 60 | 23.8% | [20, 21, 19] | 249 | 136 | 54.6% | [38, 49, 49] |
| blendupper50 | 3 | 2-4 | 249 | 82 | 32.9% | [25, 29, 28] | 253 | 139 | 54.9% | [38, 55, 46] |
| blendupper50 | pooled | 2-4 | 749 | 215 | 28.7% | | 750 | 387 | 51.6% | |
| blendupper50 | fresh | 5-7 | 245 | 69 | 28.2% | [23, 22, 24] | 247 | 122 | 49.4% | [37, 47, 38] |
| ready_upper50 | 1 | 2-4 | 251 | 251 | 100.0% | [86, 83, 82] | 249 | 249 | 100.0% | [85, 83, 81] |
| ready_upper50 | 2 | 2-4 | 252 | 252 | 100.0% | [85, 84, 83] | 247 | 247 | 100.0% | [84, 79, 84] |
| ready_upper50 | 3 | 2-4 | 251 | 250 | 99.6% | [83, 83, 84] | 252 | 252 | 100.0% | [83, 83, 86] |
| ready_upper50 | pooled | 2-4 | 754 | 753 | 99.9% | | 748 | 748 | 100.0% | |
| ready_upper50 | fresh | 5-7 | 245 | 244 | 99.6% | [81, 82, 81] | 246 | 246 | 100.0% | [81, 84, 81] |

Pooled Wilson 95% intervals on the fresh seeds (recC): base 0/242 = 0.0% (0.0-1.6%), blendupper50 122/247 = 49.4% (43.2-55.6%), ready_upper50 246/246 = 100.0% (98.5-100.0%). The fresh-seed pass was not used to choose anything and agrees with the repeats.

Every run also has 96/96 no-push controls still walking (32 per seed; 288/288 per pooled block).

## What changed and why
- **The v3 gate fires much more often.** Hand-overs per induced fall: base 248/746 (33.2%) -> 420/749 (56.1%); blendupper50 215/749 (28.7%) -> 387/750 (51.6%); ready_upper50 753/754 -> 748/748 (the readiness gate already fired for every fallen robot). Time from fall detection to hand-over falls too: base 7.44 -> 6.48 s mean.
- **The unblended baseline gets worse, not better.** With recC, 412 of 420 hand-overs re-fell (98.1%, vs 132 of 248 = 53.2% with R9hg) and faster (0.96 s vs 1.61 s mean). End-to-end success falls from 2.5% to 0.4%. The gross hand-over-state metrics barely moved (mean over the 9 runs, 128 envs each): foot spread 0.45 vs 0.44 m, leg deviation 0.29 vs 0.27 rad, torso 1.36 vs 1.25 rad, arm 1.32 vs 1.34 rad, fingers 1.27 vs 1.28 rad. So the extra robots that recC gets through the v3 gate stand in states the walking policy still cannot take over from, and the reason is not captured by these group means.
- **With the upper-body blend nothing re-falls either way** (0 of 215 with R9hg, 0 of 387 with recC), and the walk after the hand-over is slightly longer than the no-push control: 6.69 m vs 6.17 m mean along-heading (R9hg 6.17 m; control 6.17-6.25 m).
- Success by push magnitude, pooled over seeds 2-4 and the three repeats (recC; induced falls -> full successes):

| push m/s | base | blendupper50 | ready_upper50 |
|---|---|---|---|
| 2.5 | 183 -> 0 | 184 -> 121 | 185 -> 185 |
| 3.5 | 278 -> 1 | 278 -> 178 | 275 -> 275 |
| 4.5 | 288 -> 2 | 288 -> 88 | 288 -> 288 |

  The hard 4.5 m/s push is where the v3-gated variant loses most of its hand-overs: 88 of 288 (30.6%) against 178 of 278 (64.0%) at 3.5 m/s.

## Conclusion
- **Yes for the v3-gated hand-over with the upper-body blend**: 28.7% -> 51.6% pooled over the same seeds and repeats (215/749 -> 387/750; the Wilson intervals, 25.6-32.0% vs 48.0-55.2%, do not overlap) and 28.2% -> 49.4% on fresh seeds. The entire gain comes from more robots passing the v3 gate (the get-up itself), because the blend already made every hand-over walk.
- **No for the plain v3 baseline**: the more permissive get-up roughly doubles the hand-overs but almost all of them re-fall (98%), so end-to-end success drops from 2.5% to 0.4%. A better get-up on its own buys standing robots, not walking robots.
- **No measurable change for the readiness-gated variant**: it was already at the ceiling (99.9% vs 100.0%, 753/754 vs 748/748).
- The remaining lever for the official v3 hand-over share is still the hand-over state, not the get-up: the blend is what converts standing into walking.

## Media
- `media/demo_fall_getup_walk.gif` (1.1 MiB, 560 px, 10 fps, 64 colours, 11.2 s, phase labels) - the README hero. **One illustrative run, not a measurement:** a single robot, recC get-up policy, `blendupper50` hand-over, walking `it1499`, 3.5 m/s push at 1.5 s, seed 5, fixed wide camera (cropped in post; the recorder has no tracking camera), no resets. Timeline from `results/portfolio_hero1_s5.json`: push 1.50 s, fall detected 2.14 s, hand-over 10.14 s; the robot is up and standing from about 4 s but the stand gate (a 2 s supported stand) only passes at 9.14 s, so it stands in a wide stance for about 5 s before the walking policy takes over, then walks 3.5 m along its heading toward the camera. The run did not complete the protocol's 5 s walk window inside the 15 s recording (`timeout_window_incomplete`), so it is not counted as a success by the batch definition. Seeds tried for this clip: 1 (4.5 m/s: fell, never passed the gate), 2 (4.5 m/s: very slow get-up), 3 (3.5 m/s: no fall), 4 (3.5 m/s: very slow get-up), then 5 (used). An earlier 8-robot recording (seed 3, `demo_blend_try1`) supplied the previous GIF; its source is gone. `media/hero_s5_sheet.png` is the contact sheet (1, 2.2, 3.5, 5, 8, 10.2, 11.5 s). Re-make with `run_demo.sh` then `make_demo_gif.sh`; results vary run to run (the same 8-robot setup gave 4 and 2 successes in two recordings).
- `media/m2c_recC_base_seed3_16envs_0000.mp4` (1.20 MiB) and `media/m2c_recC_blendupper50_seed3_16envs_0000.mp4` (1.30 MiB), recorded exactly like the M2b videos (`--num_envs 16 --seed 3 --levels 0,2.5,3.5,4.5`): base 11 falls, 5 hand-overs, all 5 re-fell, 0 successes; `blendupper50` 11 falls, 5 hand-overs, 0 re-fell, 5 successes. Uncut 27 s, 1280x720. The `.mp4` files are gitignored by the root `*.mp4` rule and are not in the repo; only the GIF and contact sheet are committed.

## Caveats and limits
- Simulation only (Isaac Lab 3.0 / Newton-MJWarp, G1 with self-collision and Unitree torque limits); no hardware.
- DEV-style evaluation: the falls come from the dynamics, no pose sets are used, DEV/HELDOUT/TEST are not touched. The readiness gate is a hand-over readiness check, **not** the official criteria-v3 stand result; the v3 headline numbers are the `base` and `blendupper50` rows.
- The blend is an evaluation-time action override (upper-body joints to the default stance); a deployment would need it implemented as a hand-over routine.
- Three repeats are enough to see the spread but not to resolve small differences: `blendupper50` ranges 45.2-54.9% across repeats. Numbers are reported exactly as measured, all 36 runs.
- The demo clip is a single recording: it is illustrative, its counts differ from the batch, and two other demo recordings (one robot, close camera) were rejected because the pushed robot left the frame.

## Reproduce
Run the first command from `recovery/` (`run.sh` changes into the Isaac Lab folder, so paths must be absolute).
```
./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 \
    --checkpoint "$PWD/checkpoints/getup_baseline_recC_it1425.pt" --walk_checkpoint "$PWD/../checkpoints/walking_newton_it1499.pt" \
    --num_envs 128 --seed 3 --levels 0,2.5,3.5,4.5 --out /tmp/out.json physics=newton_mjwarp      # base; add --settle 50 --blend 50 --blend_joints upper for blendupper50, --trigger ready for ready_upper50
pilots/m2c_handover_recC_20261110/run.sh r1 2,3,4 base blendupper50 ready_upper50                # one repeat on one seed block
pilots/m2c_handover_recC_20261110/run_sweep.sh                                                   # the 36 runs reported here
python3 pilots/m2c_handover_recC_20261110/aggregate_m2c.py --detail --dist base blendupper50 ready_upper50
pilots/m2c_handover_recC_20261110/run_videos.sh                                                  # the two 16-robot videos
pilots/m2c_handover_recC_20261110/run_demo.sh && pilots/m2c_handover_recC_20261110/make_demo_gif.sh  # demo clip + GIF
```
`ledger.csv`: wall seconds per job (about 30 s each for 128 envs; the demo recording is about 70 s).
