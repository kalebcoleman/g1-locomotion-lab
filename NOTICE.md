# Licenses, scope and attribution

## What the BSD-3-Clause `LICENSE` of this repository covers

Only material created for this project by Kaleb Coleman: `scripts/`, `diagnostics/` (scripts and trace summaries), `README.md`, `NOTICE.md`, the figures
(`media/progression_chart.png`, `media/architecture.png`) and the result files in `results/` (evaluation JSON/CSV,
training-curve CSV).

It does **not** cover:

| Item | Where | Owner / terms |
|---|---|---|
| Resolved training configs | `configs/*.yaml` | Written by Isaac Lab; Isaac Lab BSD-3-Clause (Copyright (c) 2022-2026 The Isaac Lab Project Developers) |
| The G1 robot, rendered in every GIF/PNG under `media/` | `media/*.gif`, `media/progression_frames.png` | Unitree Robotics model (see below); the ground/scene and rendering come from Isaac Lab/Isaac Sim. These renders are shown for illustration; the underlying assets are not redistributed |
| Upstream libraries | not included | licenses below |

This repository does **not** contain Isaac Lab or Isaac Sim source, the Unitree G1 model files (USD/URDF/meshes), or any
NVIDIA asset. They are used by reference and fetched by Isaac Lab at run time.

## Upstream code and asset licenses (copies in `THIRD_PARTY_LICENSES/`)

| Component | Version | License | Copyright holder | Copy |
|---|---|---|---|---|
| Isaac Lab (task `Isaac-Velocity-Flat-G1`, env, wrappers) | 3.0.0 | BSD-3-Clause | The Isaac Lab Project Developers | `IsaacLab-BSD-3-Clause.txt` |
| Unitree G1 model (`g1_minimal.usd`, via Isaac Lab asset bucket) | as served Nov 2026 | BSD-3-Clause per Isaac Lab's `docs/licenses/assets/unitree-license.txt` | HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), 2016-2022 | `Unitree-G1-model-BSD-3-Clause.txt` |
| NVIDIA Isaac Sim | 6.2.0 | Apache-2.0 for the open-source part; additional components, 3D models and textures under the *NVIDIA Isaac Sim Additional Software and Materials License* (v. June 9, 2025), which forbids redistributing, modifying or creating derivative works of that Software (sections 2.2, 2.4) | NVIDIA | `IsaacSim-license.txt` |
| rsl_rl (PPO) | 5.5.1 | BSD-3-Clause | ETH Zurich, Nikita Rudin; NVIDIA CORPORATION & AFFILIATES, 2021 | `rsl_rl-BSD-3-Clause.txt` |
| Newton physics | 1.6.1 | Apache-2.0 | Newton developers | `Newton-Apache-2.0.txt` |
| MuJoCo-Warp | 3.12.0 | Apache-2.0 | Google DeepMind / contributors | `MuJoCo-Warp-Apache-2.0.txt` |
| MuJoCo | 3.12.0 | Apache-2.0 (file also lists bundled third-party licenses) | Google DeepMind | `MuJoCo.txt` |
| NVIDIA Warp | 1.17.0 | Apache-2.0 | NVIDIA | `NVIDIA-Warp-Apache-2.0.txt` |
| PyTorch | 2.11.0 | BSD-style | PyTorch contributors | `PyTorch.txt` |

The license texts are copied from the `docs/licenses` folder of the Isaac Lab 3.0.0 checkout used for the run.
Check each project for its current terms. PPO: Schulman et al., arXiv:1707.06347. Isaac Lab:
Mittal et al., arXiv:2511.04831.

The scripts call Isaac Lab APIs and, for argument parsing, its `play_rsl_rl` entry point; no Isaac Lab source file is
copied. If you copy Isaac Lab code into this repository later, keep its notice.

## Not resolved (see README and the review discussion)

- The scope of `unitree-isaac-sim-license.txt` in Isaac Lab's license folder (Mozilla Public License 2.0, with no
  Unitree-specific text) is unclear and it is not claimed to apply to the G1 asset used here.
- Renders (GIF/PNG) showing the G1 model and Isaac Sim ground plane: NVIDIA's Isaac Sim License FAQ
  (docs.isaacsim.omniverse.nvidia.com, "License FAQ", page dated June 2026) says that selling only simulation
  outputs (videos, reports, datasets) does not require an NVIDIA AI Enterprise license, and that NVIDIA-provided assets are
  covered by the Additional Software and Materials License. That supports publishing renders, but neither text
  explicitly addresses public posting of renders; a public NVIDIA forum question on exactly this (Aug 2026) was closed
  without a posted answer. The renders contain no asset files.
- Trained policy weights are not included in this repository.

Trademarks (NVIDIA, Isaac, Omniverse, Unitree) belong to their owners. This is an independent project, not
affiliated with or endorsed by them. Nothing here is legal advice.
