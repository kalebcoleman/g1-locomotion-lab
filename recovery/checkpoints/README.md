# Recovery checkpoints

| file | role | sha256 |
|---|---|---|
| `getup_baseline_recC_it1425.pt` | **current get-up baseline** (adopted 2026-11-09) | `a518419e347497603bc59ebe...` (full hash: `sha256sum` of the file) |
| `R9hg_it1100_previous_baseline.pt` | **previous baseline**, preserved unchanged (same bytes as the lab host's `checkpoints/R9hg_it1100_candidate.pt`) | `665a89f58b3a3d76a1319783...` |
| `laneA_v1_situp_it999.pt`, `V11_launchpen5_assist_it599.pt` | earlier recovery checkpoints | |

Both baseline files load into the task `G1-Recovery-R9hg-v0` (123-d observation, 37-d action, unchanged by the baseline change). No checkpoint file was deleted or overwritten when the baseline changed.

## The current baseline: recC
- **What it is:** the `g4j1` run, stage `b1p2`, iteration 1425 (`pilots/g4_humanlike_getup_20261108/`). Provenance chain, every step a short warm-started stage that only read its start checkpoint: `R9hg_it1100` -> `g4g1` stages (stance ramp, iterations 1125, 1175, 1249, 1323) -> `g4g1_w8p0_it1350` -> `g4j1` stage `b1p5` (it1399) -> `b1p2` (it1425).
- **Training setup that made this possible:** `env.rewards.dof_limits_body.weight=-100.0` (was -10), `agent.algorithm.learning_rate=3e-5`, stance penalty `env.rewards.stance.weight=-8.0` with `params.pelvis_h=0.12, free=0.45, full=1.5`, and hip roll and yaw targets clipped to +-1.2 rad **during the last stages**. The clip is **not** needed to run the policy: the numbers below are measured without it.
- **Why it replaces R9hg_it1100** (HELDOUT, 3 evaluations each, criteria v3 before 3b, ground-only flight, assist off):

| | previous baseline R9hg it1100 | recC |
|---|---|---|
| v3 (of 256) | 125 / 131 / 132, mean 129.3 (50.5%) | 176 / 174 / 174, mean 174.7 (68.2%) |
| by direction (side_a / prone / side_b / supine), mean | 31 / 47 / 19 / 33 | 52 / 52 / 36 / 35 |
| hard-limit step fraction | 0.0116 | 0.0012 |
| soft-limit step fraction | 0.039 | 0.010 |
| peak foot spread, pelvis above 0.12 m | 1.38 m | 1.39 m |
| DEV v3 | 92 | 150 |
3b (self-penetration): 236 of 236 robots examined are within the 3% rule for recC (median worst depth 3.4 mm). HELDOUT was never used to choose the checkpoint; TEST untouched.
- **What it does not do:** it does not reduce the straddle (peak spread unchanged), so it is a better get-up, not a more human-like one. The weakest directions are side B and supine. One training lineage and one seed. Details and all 22 runs: `pilots/g4_humanlike_getup_20261108/README.md`.

## Reproducibility note
The M2 and M2b evaluation scripts (`pilots/m2_fall_handover_20261108/`, `pilots/m2b_handover_fixes_20261108/`) still point at `checkpoints/R9hg_it1100_candidate.pt`, because their recorded results were measured with that policy. They are left unchanged on purpose; to repeat them with the new baseline, pass `recovery/checkpoints/getup_baseline_recC_it1425.pt` as `--checkpoint`. No script in the repo has a default recovery checkpoint.
