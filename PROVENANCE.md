# Provenance of imported components (integration project)
Status 2026-11-08: this history was written as a private, local-only integration repo (branch `integration/main`) and was published to this repository through PR #4 on Kaleb's decision. The table below records the state at import time; entries marked private were private then. Originals are untouched.
| Component | Source | Commit | Visibility | How imported | Licence |
|---|---|---|---|---|---|
| `walking/` | https://github.com/kalebcoleman/g1-locomotion-lab (local `projects/g1-locomotion-lab`) | 8f137ed | public | `git subtree add` with full history | BSD-3 (original) + third-party notices inside |
| `recovery/` | https://github.com/kalebcoleman/g1-recovery (local `projects/g1-recovery-export`) | 80f8d59 | private | `git subtree add` with history | BSD-3 (original) + NOTICE |
| not imported (history only referenced) | `projects/g1-recovery` working repo, branch feat/recovery-base @ 12380f2: full experiment history, 329 run artifacts, logs, videos (605 MB) | | local | referenced; the export is its curated subset | |
| not imported (referenced) | `projects/g1-tasks` branch feat/g1-newton @ 2bf6d8f: the original walking experiment (raw MP4s, PhysX comparison, original results) | | local | referenced; its curated, Newton-only subset is `walking/` | |
| `checkpoints/walking_newton_it1499.pt` | `envs/lab-3.0.0-src/logs/rsl_rl/g1_flat/2026-11-06_15-13-37/model_1499.pt` (Newton run, seed 1, 1024 envs, 1500 its) | sha256 daa544a8d3c009cbd85e9ed085d32bb56d820995a9cf243cfcf6efe4e09f6fa6 | private copy | file copy | trained here |
| `checkpoints/laneA_v1_situp_it999.pt`, `V11_launchpen5_assist_it599.pt` | recovery export | | private | in `recovery/checkpoints` | trained here |
Isaac Lab 3.0.0 (commit e3f7d9f, installed in `envs/lab-3.0.0`, source `envs/lab-3.0.0-src`) is a dependency, not part of this repo.
No private material (recovery code, checkpoints, history, media) may go into the public `g1-locomotion-lab` repo until Kaleb reviews the exact public contents.
