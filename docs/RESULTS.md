# Results

Every number here comes from the write-ups linked in each section. **All results are simulation only** (Isaac Lab 3.0.0, Newton / MuJoCo-Warp, Unitree G1) and, unless stated otherwise, come from **one training seed and one run** — the evaluation itself is not bit-repeatable, so treat counts as samples, not exact values. Back to the [overview](../README.md).

**How to read the get-up numbers:** all get-up figures use criteria v3 *before* the self-penetration check (3b), on the **HELDOUT** set (256 poses, fixed comparison set, never used for training or checkpoint selection). The **TEST** set is untouched. Checkpoints are chosen on **DEV**, which is a development set, not held-out. Each final candidate was evaluated three times; the means below are over those three evaluations. Detail: [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md).

## Walking (24 s pushed test)

| | |
|---|---|
| Survivors, final checkpoint (iteration 1499) | **237 / 256** |
| Reruns, same checkpoint, same command | 231 / 256 and 233 / 256 |
| Falls by phase (stand / walk / turn / stop) | 0 / 2 / 17 / 0 — most falls happen while turning |
| Test | 24 s: stand, walk 0.8 m/s, turn 0.8 rad/s, stop; random pushes up to 0.5 m/s every 2.5 s |
| Training-log episode length at iteration 1499 | 971 of a possible 1,000 steps (separate metric, not the scripted test) |

One training seed (seed 1), one evaluation per checkpoint, no confidence intervals. Full table, method and rerun caveats: [docs/WALKING.md](WALKING.md#results-at-a-glance).

## Getting up (criteria v3, HELDOUT, assist off)

| | previous baseline (R9hg it1100) | current baseline (recC) |
|---|---|---|
| Pass rate of 256 poses, mean of 3 evaluations | 129.3 (**50.5%**), runs 125 / 131 / 132 | 174.7 (**68.2%**), runs 176 / 174 / 174 |
| By direction, of 64 each (side A / prone / side B / supine) | 31 / 47 / 19 / 33 | 52 / 52 / 36 / 35 |
| Hard-limit step fraction | 0.0116 | 0.0012 |
| Peak foot spread while rising | 1.38 m | 1.39 m |

Self-penetration (3b): 236 of 236 examined robots are within the 3% rule for recC (median worst depth 3.4 mm), so full v3 equals the figure above. One training lineage, one seed; side B and supine remain the weakest directions. What "passes" means: supported 2 s stand, no launch or flight, joint, torque and self-penetration limits, normal stance (feet ≤ 0.60 m), still up at the end, no assist — [recovery/README.md](../recovery/README.md), [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md). Write-up: [G4](../pilots/g4_humanlike_getup_20261108/README.md), checkpoint provenance: [recovery/checkpoints/README.md](../recovery/checkpoints/README.md).

## Hand-over (fall → get-up → walk)

| Variant | v3-gated end-to-end success |
|---|---|
| Hand over to walking immediately (M2) | 6 of 250 induced falls (2.4%) |
| Baseline behaviour re-run (M2b, pooled over seeds 2–4) | 19 of 746 (2.5%) |
| Settle 1 s + upper-body blend, v3 gate | **215 of 749 (28.7%)** pooled; 28.2% on fresh seeds 5–7; **0 re-falls in every run** |
| **Settle 1 s + upper-body blend, v3 gate, recC get-up (M2c)** | **387 of 750 (51.6%, 95% CI 48.0–55.2%)** pooled; 49.4% on fresh seeds 5–7; 0 re-falls |
| Readiness gate + settle + blend | 753 of 754 (99.9%) pooled — **not the same test as the v3 stand gate**: the readiness gate has no validity clauses, and at hand-over the official v3 gate was true for only 1–3 robots |

Evaluation only — no training, no weight changes; rows without "recC" use the *previous* recovery checkpoint (R9hg it1100). With recC (M2c) the v3 gate fires for 56% of falls instead of 33%; without the blend 98% of those hand-overs re-fall (0.4% success), with it none do. With recC the readiness variant reached 748 of 748. Pushes of 2.5 / 3.5 / 4.5 m/s, 128 envs per seed, seeds 2–4 plus fresh seeds 5–7. Run-to-run spread matters: the baseline's three runs gave 0.8%, 2.4% and 4.4%. Write-ups: [M2](../pilots/m2_fall_handover_20261108/README.md), [M2b](../pilots/m2b_handover_fixes_20261108/README.md), [M2c](../pilots/m2c_handover_recC_20261110/README.md).

## Pushes (walking policy, friction 1.0)

- **M50 (push at which half the standing robots fall): 2.16 m/s** logistic fit [95% bootstrap CI 2.08, 2.24]; 2.02 m/s interpolated. Training used random pushes up to **0.5 m/s**, so this is about **4× the training push**.
- Survival pooled over directions: 97% at 1 m/s, 74% at 1.5, 52% at 2, 6% at 4 (Wilson intervals in the write-up).
- By direction: weakest to the left (M50 1.43 / 1.48 m/s) and rear-left (1.35 / 1.40), strongest forward-right (3.62 / 3.44) and forward (2.58 / 2.85). Cells carry 11–12 robots at 2.0 m/s, so single directions have wide uncertainty.

Seeds 1–3, one push per episode at 3.0 s, 8 directions. Write-up: [M4](../pilots/m4_robustness_20261108/README.md).

## Low friction

| floor and body friction | walked the first 3 s | M50 |
|---|---|---|
| 1.0 (training) | 1421/1440 (99%) | 2.16 m/s |
| 0.4 | 1416/1440 (98%) | 1.68 m/s |
| 0.3 | 1330/1440 (92%) | 1.45 m/s |
| 0.2 | 806/1440 (56%) | 0.96 m/s (few robots still standing; unreliable) |

Friction at or above 0.4 changes little; below that the policy breaks down quickly. Note the solver quirk: contacts use the *larger* of the floor and body friction values, so both must be lowered together (the first sweep that lowered only the floor was invalid and discarded). Write-up: [M4](../pilots/m4_robustness_20261108/README.md).

## Terrain (blind, flat-trained policy, 10 s cells, seeds 1–3)

Hardest level with at least 80% survival (every easier level must pass too):

| Terrain | Hardest level |
|---|---|
| Slope down | 19° |
| Slope up | 9° (but survivors cover only 2.56 m in 10 s) |
| Steps down | 5.8 cm |
| Steps up | 2.8 cm — **all 24 robots fell at 4.3 cm** and at every larger height (0/24 at levels 1–7) |
| Random ground | 8 cm noise (92%; falls to 50% at 10 cm, 4% at 14 cm) |

24 robots per cell (8 per seed × 3 seeds), 95% Wilson intervals are wide (e.g. 24/24 = [86, 100]). The policy was trained on flat ground and receives no height scan. Write-up: [M3](../pilots/m3_terrain_baseline_20261108/README.md).

## Statistics notes

- Confidence intervals are Wilson score intervals for binomial survival rates and percentile bootstrap for the M50 logistic fits; they cover sampling noise, not simulation nondeterminism.
- Robust claims need ≥ 3 training seeds; nothing here meets that bar — it is all one seed per policy ([EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md)).
- Evaluation repeats of one checkpoint varied by 3–8 of 256; the walking test's three samples span 231–237 of 256.