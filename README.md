# G1 recovery (get-up) experiments: Isaac Lab / Newton-MJWarp

Experiment project (simulation only): train a Unitree G1 to get up from the ground **unassisted and physically plausibly**, then chain to a walking policy.
- **Read first:** [`REPORT.md`](REPORT.md) (results, lessons, profiling), [`CRITERIA.md`](CRITERIA.md) (success definitions frozen before evaluation), [`STATUS.md`](STATUS.md) (run log, commands, next steps).
- Code: `g1_recovery/` (task package: env, pose curriculum, success tracker, assist, lane rewards), `scripts/` (train/eval/record/diagnose, GPU launch logger), `research/` (HoST, HumanUP, APEX assessments).
- Data: `results/` (per-run eval/diagnostic JSON, training summaries, curves), `heldout_poses.npz`, `bank_v1.npz`, two small checkpoints in `checkpoints/`, selected media in `media/`.
- Honest status: first plausible unassisted recoveries are only 2/256 held-out poses (single seed); an earlier 256/256 was a simulator launch exploit and is rejected. See REPORT.md.
Needs Isaac Lab 3.0.0 (not included). Licences and credits: [`NOTICE.md`](NOTICE.md), BSD-3-Clause for original material.
