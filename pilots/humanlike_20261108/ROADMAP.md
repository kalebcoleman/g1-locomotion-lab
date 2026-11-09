# G1 human-like get-up: staged roadmap (lab thread, simulation only)
Rules: existing scripts/config only until a source change is approved; no hardware, no paid services, no merge, nothing counted as published until a commit/PR exists. Baseline `R9hg it1100` stays the reference; the failed pilot is a negative result, not a base to continue.

| stage | goal | state |
|---|---|---|
| M0 | reproducible commands, provenance, resume/checkpoint/gate validation, document negative pilot | done; committed on branch `m1-metrics-and-m0-docs` (local, not pushed) |
| M1 | movement-quality telemetry + early-rise spread knobs | implemented and validated on the frozen baseline (`eval_recovery.py`, `diag_trace.py`, `metrics.movement_metrics`, `lane_a.STANCE_DEFAULTS`); committed on the branch, not pushed |
| M2 | frozen walk -> actual fall -> get-up -> stable stand -> walk, resets never credited | implemented as an evaluation (`eval_fall_handover.py`); 6 of 250 induced falls complete the sequence under the v3 gate, 25 of 248 under a strict-stand gate; see `pilots/m2_fall_handover_20261108/README.md` |
| M3 | rough terrain, same 37-joint morphology | baseline measured (evaluation only, blind walking, 123-d observation unchanged): hardest level with >= 80% survival is 19 deg (slope down), 9 deg (slope up), 5.8 cm (steps down), 2.8 cm (steps up), 8 cm noise (rough ground); see `pilots/m3_terrain_baseline_20261108/README.md` |
| M4 | pushes, friction randomisation | measured (evaluation only): push M50 = 2.16 m/s [2.08, 2.24] at friction 1.0, falling to 1.68 m/s at 0.4 and below that quickly; see `pilots/m4_robustness_20261108/README.md` |
| imitation | human reference tracking | gated: legal data (LAFAN1 NC-ND), morphology (no waist roll/pitch in this env), trainer (none usable); see `REFERENCE_DATA_RESEARCH.md` |

## M0 evidence (CPU-only, read-only; files in `m0/`)
- Checkpoint contents (baseline, both pilot checkpoints, walking it1499): `actor_state_dict` (9 keys incl. `distribution.std_param`), `critic_state_dict` (8), `optimizer_state_dict`, `iter`, `infos` (= None). No observation-normalizer keys; actor input 123, output 37. Curriculum level is not saved.
- Pilot resume mode, from saved artifacts (`m0/run_manifest_humanlike_p1_s21.json`): **iteration counter restored (1100 to 1199), actor+critic loaded, optimizer RESET** (Adam step 2020 in the baseline checkpoint vs 20 at the first save of the new run; 2000 at the last). So it was a warm start, not a full resume. Overrides requested are all present in the saved `env.yaml`/`agent.yaml`; learning rate fixed at 3e-4.
- Driver failure, exact: the generated `driver_train.sh` has `LAB=` (empty; my heredoc expanded an unset shell variable), so run-folder lookup used `/logs/...`, returned nothing, and `2>/dev/null` hid it. The it1150 DEV gate and checkpoint copy were skipped, and the following evals ran on a non-existent `ckpt/cand_final_it.pt` (4 jobs rc=1 in `ledger.csv`) without aborting the sequence. Training itself was unaffected.
- Gate tests, reproducible: `m0/test_gates.sh` (CPU, ~15 s). Cases: good baseline passes; missing file exit 2; wrong sha exit 3; no run dir exit 4; bad structure exit 5; wrong obs dim exit 6; regression for the empty-root bug exit 4; real pilot run dir found. All pass (last run: 8/8).
- `m0/driver_v2_template.sh`: fail-loud sequence (set -euo pipefail, explicit args, inventory and run-dir gates). Not run. `gpu.sh` still swallows exit codes (needs `exit $rc`); that and the template are local tooling, untracked.

## Exact existing commands
See `HANDOFF.md` ("Reproduce"). True full resume with the existing trainer: `--checkpoint <ckpt>` **without** `--no-resume_optimizer_state` (not exercised here). Warm start as used: add `--no-resume_optimizer_state`.

## Source changes made on the branch
1. `g1_recovery/lane_a.py`: `stance_excess` params declared on both RewTerms (`STANCE_DEFAULTS`, equal to the function defaults), so `env.rewards.stance.params.{free,full,pelvis_h}` can be overridden.
2. `g1_recovery/metrics.py` (`movement_metrics`), `scripts/eval_recovery.py` (`movement` block), `scripts/diag_trace.py` (hip, action-rate, acceleration traces); 5 new unit tests.
3. Not done: Isaac MCP cannot see `g1_recovery` tasks; hip-splay term; continuous fall runner (M2); reference tracking.
4. Tracked-repo state at branch point: HEAD `330493f` on `integration/main`; `pilots/` and the baseline copy in `checkpoints/` were untracked.
