# References: humanoid get-up / fall recovery (checked 2026-11-07)

Licence readings are ours, not legal advice. Nothing here is vendored; ideas are re-implemented with attribution. Earlier assessments: `host_humanup_vs_ours.md`, `apex_standup_assessment.md`, `lane_b_references.md`.

## Code

| project | link | licence | stack | notes |
|---|---|---|---|---|
| HoST (Humanoid Standing-up Control) | https://github.com/InternRobotics/HoST | MIT | Isaac Gym, legged_gym, 23-DoF G1 | Paper arXiv:2502.08378, RSS 2025; project page https://taohuang13.github.io/humanoid-standingup.github.io/ ; PDF https://www.roboticsproceedings.org/rss21/p064.pdf |
| HumanUP (Learning Getting-Up Policies) | https://github.com/RunpeiDong/HumanUP | Apache-2.0 | Isaac Gym, G1 | Paper arXiv:2502.12152, RSS 2025; page https://humanoid-getup.github.io/ |
| HoST-pytorch | https://github.com/lucidrains/HoST-pytorch | MIT | PyTorch only | algorithm reference, no simulator |
| g1-humanoid-rl | https://github.com/chaoticcuriosity-io/g1-humanoid-rl | MIT | mjlab (MuJoCo-Warp), G1 | Chapter 14 get-up from scratch: https://github.com/chaoticcuriosity-io/g1-humanoid-rl/blob/main/docs/reports/14-building-get-up-from-scratch.md ; task source in `recovery-task/` |
| mjlab | https://github.com/mujocolab/mjlab | Apache-2.0 | MuJoCo-Warp, Isaac-Lab-style manager API | hosts the BeyondMimic tracking task now; G1 with Unitree actuator limits |
| BeyondMimic / whole_body_tracking | https://github.com/HybridRobotics/whole_body_tracking | MIT | Isaac Lab 2.1 (PhysX), G1 | retired 2026-10-16, points to mjlab; paper arXiv:2508.08241; site https://beyondmimic.github.io/ |
| unitree_rl_lab | https://github.com/unitreerobotics/unitree_rl_lab | BSD-3 | Isaac Lab, G1-29dof velocity and mimic | official actuator parameters |
| unitree_rl_mjlab | https://github.com/unitreerobotics/unitree_rl_mjlab | BSD-3 (per Unitree) | MuJoCo | official |
| FRASA (Fall Recovery And Stand-up Agent) | https://github.com/Rhoban/frasa | MIT | MuJoCo, stable-baselines3 / sbx, Sigmaban 20-DoF | paper arXiv:2410.08655; 2 s settle, velocity-mode actions, upside-down termination |
| APEX-humanoid | https://github.com/apex-humanoid/APEX-humanoid | BSD-3 (Isaac Lab text) | Isaac Lab 1.3.0 / PhysX fork, 29-DoF G1 | `Isaac-StandUp-G1`, prone-only start; checkpoint/USD have no separate licence |

## Papers

| paper | link | notes |
|---|---|---|
| Huang et al., Learning Humanoid Standing-up Control across Diverse Postures (HoST), RSS 2025 | https://arxiv.org/abs/2502.08378 | force curriculum essential; supine+prone joint training interfered |
| He, Dong et al., Learning Getting-Up Policies for Real-World Humanoid Robots (HumanUP), RSS 2025 | https://arxiv.org/abs/2502.12152 | two-stage discovery then tracking of own rollouts |
| Xu, Li, Lin, Yu, Unified Humanoid Fall-Safety Policy from a Few Demonstrations (FIRM), Nov 2025 | https://arxiv.org/abs/2511.07407 | G1 23-DoF, sim and real; no code found; project page https://firm2025.github.io/ |
| Gaspard et al., FRASA, 2024 | https://arxiv.org/abs/2410.08655 | kid-size humanoid, end-to-end RL |
| Liao et al., BeyondMimic, 2025 | https://arxiv.org/abs/2508.08241 | motion tracking on G1 |
| Harvey et al., Robust Motion In-betweening (LAFAN1), SIGGRAPH 2020 | https://github.com/ubisoft/ubisoft-laforge-animation-dataset | CC BY-NC-ND 4.0; source of the Unitree G1 retargets |

## Data

| dataset | link | licence | notes |
|---|---|---|---|
| Unitree LAFAN1 Retargeting Dataset (G1) | https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset | BSD-3 file, but upstream LAFAN1 is CC BY-NC-ND 4.0 | six `fallAndGetUp*` clips; private non-commercial use only, never redistribute or replay in published media |

## Frameworks used here

| item | link | licence |
|---|---|---|
| Isaac Lab 3.0.0 | https://github.com/isaac-sim/IsaacLab | BSD-3 |
| Newton | https://github.com/newton-physics/newton | Apache-2.0 |
| MuJoCo Warp | https://github.com/google-deepmind/mujoco_warp | Apache-2.0 |
| rsl_rl | https://github.com/leggedrobotics/rsl_rl | BSD-3 |
| Unitree G1 description | https://github.com/unitreerobotics/unitree_ros | BSD-3 |
