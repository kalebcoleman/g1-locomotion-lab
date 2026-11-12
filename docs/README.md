# Documentation index

| Document | What it covers |
|---|---|
| [RESULTS.md](RESULTS.md) | Every headline number with its source, seeds and caveats |
| [WALKING.md](WALKING.md) | The walking policy: method, results, reproduction, limitations |
| [SETUP.md](SETUP.md) | Isaac Lab setup and the commands to run training and evaluation |
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | What this project added to Isaac Lab, file by file |
| [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md) | The get-up evaluation criteria (v3), pose sets and amendments, frozen before scoring |
| [pilots/README.md](../pilots/README.md) | Index of the dated experiments, each with its own write-up |

## Glossary

- **R9hg, recC.** Names of get-up policies. `R9hg` is also the name of the training task (`G1-Recovery-R9hg-v0`). The checkpoint `R9hg_it1100` is the *previous* baseline: it passes the full checks on about 50% of held-out poses. `recC` (run `g4j1` stage `b1p2`, iteration 1425) is the *current* baseline: 68%. Both files are in [`recovery/checkpoints/`](../recovery/checkpoints/README.md).
- **Criteria v3.** The validity checks for a get-up: a 2 s supported stand with no launch or flight, joint and torque limits respected, a normal stance, and still up at the end ([EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md)). Results labelled "v3 before 3b" are scored without the last check.
- **3b.** The self-penetration check: on no more than 3% of sampled frames may two non-neighbouring body parts overlap by more than 2 cm. It runs offline on a recorded trace.
- **DEV, HELDOUT, TEST.** Three fixed sets of 256 lying poses. DEV is for iterating, HELDOUT for comparing finished policies, TEST is untouched until one final claim. `recovery/heldout_poses.npz` is the original pose set, byte-identical to `recovery/sets/dev_v1.npz`; despite its name it is the DEV set, not the HELDOUT set, and it was not renamed so that older write-ups and scripts still resolve.
- **M50.** The push strength (added speed, m/s) at which half the robots that were standing at the push fall. Reported with a logistic fit and a bootstrap interval.
- **Pilots.** Short, bounded experiments, each with a README:
  - **M2**: chain the walking and get-up policies through an induced fall ([write-up](../pilots/m2_fall_handover_20261108/README.md)).
  - **M2b**: fix the hand-over (settle, upper-body blend, readiness gate) ([write-up](../pilots/m2b_handover_fixes_20261108/README.md)).
  - **M2c**: repeat M2b with the current get-up policy, recC ([write-up](../pilots/m2c_handover_recC_20261110/README.md)).
  - **M3**: blind terrain robustness of the walking policy ([write-up](../pilots/m3_terrain_baseline_20261108/README.md)).
  - **M4**: push and friction robustness ([write-up](../pilots/m4_robustness_20261108/README.md)).
  - **G4**: training a less wide, more human-like get-up; the target was not met, but it produced recC ([write-up](../pilots/g4_humanlike_getup_20261108/README.md)).
