# Provenance of imported components
This repository combines two projects that were developed separately: the walking project and the get-up (recovery) project. Their histories were merged here by `git subtree add`, so the commits of both are preserved. The table records where each part came from; entries marked private were private at import time.

| Component | Source | Commit | Visibility | How imported | Licence |
|---|---|---|---|---|---|
| `walking/` | the original public walking repository (this repository's earlier history) | 8f137ed | public | `git subtree add` with full history | BSD-3 (original) + third-party notices inside |
| `recovery/` | a private development repository for the get-up experiments (curated export) | 80f8d59 | private | `git subtree add` with history | BSD-3 (original) + NOTICE |
| not imported (history only referenced) | the full get-up working repository: complete experiment history, run artifacts, logs and videos | | private | referenced; `recovery/` is its curated subset | |
| not imported (referenced) | the original walking experiment (raw MP4s, PhysX comparison, original results) | | private | referenced; its curated, Newton-only subset is `walking/` | |
| `checkpoints/walking_newton_it1499.pt` | Newton run, seed 1, 1024 envs, 1500 iterations (`model_1499.pt`) | sha256 daa544a8d3c009cbd85e9ed085d32bb56d820995a9cf243cfcf6efe4e09f6fa6 | public | file copy | trained here |
| `recovery/checkpoints/laneA_v1_situp_it999.pt`, `V11_launchpen5_assist_it599.pt` | get-up export | | public | in `recovery/checkpoints` | trained here |

Isaac Lab 3.0.0 (commit e3f7d9f) is a dependency, not part of this repository.
