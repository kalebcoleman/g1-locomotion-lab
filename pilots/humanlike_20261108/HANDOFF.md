# Pilot handoff: reward-regularised warm-start from R9hg it1100 (simulation only)

**Result: negative.** The pilot lowered success and did not change the early-rise leg flare. At the time of writing nothing here had been committed or pushed.
Paths are relative to `projects/g1-locomotion-lab-integrated/` (base commit `330493f`; no source edits were made). Run folder `pilots/humanlike_20261108/`. Run date 2026-11-08, MST.
Companion files: `IMPLEMENTATION_SPEC.md` (what to build, acceptance checks), `REFERENCE_DATA_RESEARCH.md` (imitation data route), `METHOD_NOTES.md` (citations).

## Method (honest label)
- **Reward-only.** Not imitation, not AMP, not reference tracking: no licensed G1 floor-getup reference was usable (see research file).
- Warm start from R9hg it1100 (fresh optimizer), task `G1-Recovery-R9hg-v0`, four reward-weight overrides, one seed. Weights were set by judgement before training, not tuned.

| setting (Hydra override) | baseline R9hg | pilot |
|---|---|---|
| `env.rewards.stance.weight` (foot-distance) | -5 | -8 |
| `env.rewards.action_rate.weight` | -0.01 | -0.03 |
| `env.rewards.dof_acc.weight` | -2.5e-7 | -1e-6 |
| `env.rewards.dof_limits_body.weight` | -10 | -15 |
| `agent.save_interval` | default | 10 |

## Provenance
| item | value |
|---|---|
| baseline | `checkpoints/R9hg_it1100_candidate.pt`, sha256 `665a89f58b3a3d76a131...`; read-only copy `ckpt/R9hg_it1100_BASELINE_frozen.pt` |
| pilot it1150 | `ckpt/cand_it1150.pt`, sha256 `25d65fab3bd5627ce0aa...` |
| pilot final (it1199) | `ckpt/cand_final_it1199.pt`, sha256 `2ab776104b1ab7bb00bf...` |
| training | seed 21, 16384 envs, 100 PPO iterations (1100 to 1199), Isaac Lab 3.0.0 `newton_mjwarp`, wall 247 s (~2.2 s/iter) |
| criteria | v3 (`EVAL_PROTOCOL.md`); headline = `amendment3.ground_only_flight.v3_before_selfpen`. 3b (self-penetration) **not run** for the pilot checkpoints |
| sets | DEV `recovery/sets/dev_v1.npz` (tuning); HELDOUT `recovery/sets/heldout_v2.npz` (comparison); TEST untouched |
| eval | 256 poses (64 per direction), 10 s, assist off, env seed 1, one evaluation per label unless noted |

## Results (HELDOUT, criteria v3 before 3b, ground-only flight; of 256)
| checkpoint | v3 | side_a | prone | side_b | supine | soft-limit rule fails* | hard-limit step frac | final foot spread (m) | time spread > 0.45 m |
|---|---|---|---|---|---|---|---|---|---|
| baseline, eval 1 | 131 | 32 | 45 | 21 | 33 | 6 | 0.012 | 0.427 | 0.40 |
| baseline, eval 2 | 122 | 30 | 44 | 19 | 29 | 4 | 0.012 | 0.428 | 0.40 |
| pilot it1150 | 64 | 20 | 13 | 15 | 16 | 136 | 0.011 | 0.403 | 0.26 |
| pilot final | 83 | 26 | 24 | 19 | 14 | 124 | 0.019 | 0.395 | 0.20 |

\*robots with position-limit violation on more than 10% of steps (criterion 4). Strict supported stand is 256/256 for every row; the drop is in the v3 sub-rules. DEV: baseline 104, it1150 40, final 56.
Historical baseline numbers (not from this run): 127, 128, 129 (`EXPERIMENT_LOG.md`). Two baseline evals today gave 131 and 122, so evaluation variation is at least 9/256; the pilot's drop is far outside it.

- **Early-rise flare unchanged.** Peak foot spread after the pelvis passes 0.40 m: 1.32 m (baseline, both pilot checkpoints). Median spread rises from 0.16 m to ~1.3 m at t≈0.35 to 0.5 s with the pelvis at 0.3 to 0.45 m, then falls to ~0.45 m by 1 s. See `media/compare_plots.png`.
- The pilot narrowed the final stance by about 3 cm (0.43 to 0.40 m). That is the only style change measured.
- Why success dropped: not diagnosed. The measured change is the soft-limit rule (above). I did not test which reward change causes it.
- The planned DEV stop check at it1150 did not run (driver bug: it could not find the run folder). With DEV 40 vs baseline 104 it would have stopped training; training had already completed. I ran all evaluations by hand afterwards.
- **Bounded-target control (no training, baseline checkpoint, hip roll and yaw targets clipped, HELDOUT):**

| clip (rad) | v3 | strict | hard-limit step frac | peak spread (m) | final spread (m) |
|---|---|---|---|---|---|
| none | 131 | 256 | 0.012 | 1.32 | 0.43 |
| +-0.50 | 46 | 248 | 0.072 | 1.14 | 0.50 |
| +-0.30 | 6 | 113 | 0.410 | 0.98 | 0.61 |

This clips a policy not trained with the clip; it does not test training with bounds.

## Media (all under `pilots/humanlike_20261108/media/`; local files, no URLs exist)
Matched renders, uncut 10 s, same first 16 HELDOUT poses (all four directions: 4 side_a, 3 prone, 4 side_b, 5 supine), same camera and seed, no selection among poses.
- `base_heldout16_0000.mp4`, `cand1150_heldout16_0000.mp4`, `candfinal_heldout16_0000.mp4`
- contact sheets `*_heldout16_sheet.png` (t = 0.5, 1, 2, 4, 6, 9 s), `compare_plots.png` (spread, pelvis height, success by direction)
- Caveat: the recorder reports the legacy stand flag, which is 16/16 for all three. By criteria v3 the same 16 poses score 10 and 6 (baseline evals 1 and 2), 4 (it1150), 5 (final) in the 256-env evals (`per_env`, indices 0 to 15). The videos therefore show that all stand, not which meet v3; failures are the v3-failing indices, not separately rendered.
- Visual: at 0.5 s every start, baseline and pilot, is already in a wide-legged posture; by 1 s all are upright. I see no visible style difference between baseline and pilot in the sheets.

## Reproduce
```
# from recovery/ ; <ckpt> is any checkpoint above
./scripts/run.sh eval_recovery.py --task G1-Recovery-R9hg-v0 --checkpoint <ckpt> --num_envs 256 --heldout sets/heldout_v2.npz --out <out.json> physics=newton_mjwarp
./scripts/run.sh diag_trace.py    --task G1-Recovery-R9hg-v0 --checkpoint <ckpt> --num_envs 256 --heldout sets/heldout_v2.npz --out <trace.npz> physics=newton_mjwarp
python pilots/humanlike_20261108/analyze_trace.py <trace.npz> <spread.json>      # interim spread metrics
# training (shell; needs PYTHONPATH=<recovery>, --external_callback=g1_recovery.register.register):
pilots/humanlike_20261108/train_pilot.sh <run_name> 16384 100 21                 # reads overrides.txt
# videos: ./scripts/run.sh record_recovery.py --task G1-Recovery-R9hg-v0 --checkpoint <ckpt> --out_dir <dir> --label <l> --num_envs 16 --heldout sets/heldout_v2.npz physics=newton_mjwarp
# sheets: ./scripts/contact_sheet.sh <mp4> <png>
```
Raw results: `results/*_eval.json`, `*_trace.npz`, `*_spread.json`, `compare_latest.json`; logs `logs/`; manifest `manifest.json`; GPU ledger `ledger.csv` (about 14 min summed wall time of all jobs, against the 2 h cap).

## M1 validation (frozen baseline, HELDOUT, 256 poses, brief eval; no training)
- Defaults unchanged: eval with the new code gives v3 (ground-only flight) **130** (side_a 35, prone 44, side_b 19, supine 32), strict 256, hard-limit step frac 0.0115, soft 0.039. Earlier baseline evals: 131, 122 (this run folder), 127 to 129 (historical).
- Override accepted: `env.rewards.stance.params.pelvis_h=0.15 env.rewards.stance.params.free=0.30` runs (v3 133). This shows the key exists; an eval does not test the reward's effect on training.
- New metrics on the frozen baseline (mean over 256 robots; median in brackets):

| metric | value |
|---|---|
| peak foot spread, pelvis > 0.12 m (m) | 1.378 (1.382) |
| peak foot spread, pelvis > 0.40 m (m) | 1.320 (1.328) |
| time of peak (s) | 0.40 (0.36) |
| final foot spread, last 1 s (m) | 0.427 (0.418) |
| share of gated steps with spread > 0.45 m | 0.385 (0.101) |
| hip roll / yaw peak abs (rad) | 1.29 / 1.26 |
| hip roll / yaw final abs (rad) | 0.141 / 0.252 |
| action rate, mean abs delta per step (rising window) | 0.540 |
| joint acceleration RMS / p99 (rad/s^2, rising window) | 168 / 628 |
| joint jerk p99 (rad/s^3, rising window) | 26,700 |

The peak foot-spread numbers match the interim script (1.320 m, 0.427 m). Hip peaks include the lying start pose. Acceleration is a finite difference at the 50 Hz control step, so it under-reads shorter spikes; no earlier numbers exist to compare. Per-direction values are in `movement.by_category` of the eval JSON.
- Tests: `recovery/tests/test_metrics.py` 19 pass (5 new); `m0/test_gates.sh` 8/8.

## Limitations
- One seed, one pilot, one evaluation per pilot checkpoint. Baseline range from two evals (122, 131) plus three historical (127 to 129).
- The pilot comparison above predates the M1 metrics, so smoothness and hip values exist only for the baseline (M1 validation). The derived thigh-splay angle in `*_spread.json` has unverified sign conventions and is not used.
- "Human-like" is not measured by any metric. Foot spread is a proxy; the videos show no clear change.
- 3b self-penetration not run for the pilot (the pilot failed anyway).

## Still open (source/config)
Done on this branch: stance params overridable; spread/hip/smoothness metrics. Still open: hip-splay term (only after the metrics show where the spread occurs), early-rise rerun, continuous fall runner (M2), reference tracking (blocked, see research).

## Publication notes
- Add a short dated entry to `EXPERIMENT_LOG.md`; keep historical R9hg numbers labelled historical; no README rewrite.
- Do not commit `*.npz` traces, checkpoints or motion data; use the repo's media convention for the mp4/png. Absolute host paths and the user directory name do not appear in this file.
