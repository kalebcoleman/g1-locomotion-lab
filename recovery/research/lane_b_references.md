# Lane B – reference sources for G1 get-up (research, 2026-11-06)

Method: read-only; GitHub/HF APIs + raw files + arXiv abstracts. Only one small CSV pulled, to /tmp/lb (nothing in repo).
Not legal advice – licence readings below are mine; confirm before publishing anything derived from data.

## 1. Motion data

### 1a. Unitree "LAFAN1 Retargeting Dataset" (G1) – the only public G1 get-up mocap retarget found
- URL: https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset (also mirrored as unitreerobotics/..., which returned an auth error to me). Last modified 2025-04-06, ~19.8k downloads.
- Get-up clips (g1/*.csv, six of 40): fallAndGetUp1_subject1/4/5, fallAndGetUp2_subject2/3, fallAndGetUp3_subject1 (1.0–1.7 MB each).
- Format (README + inspected fallAndGetUp1_subject1.csv): no header, 36 cols/row = root xyz + quat XYZW (7) + 29 joints; **30 FPS**; 5047 rows (~168 s) in that file, so a clip holds several fall/get-up episodes and must be segmented. Root z ranges 0.05–0.80 m (genuinely on the floor).
- Joint order: L leg(hip pitch/roll/yaw, knee, ankle pitch/roll), R leg, waist yaw/roll/pitch, L arm (sh pitch/roll/yaw, elbow, wrist roll/pitch/yaw), R arm. Names match Unitree G1-29dof URDF/Isaac Lab names; no fingers.
- README caveat: retargeting is kinematic only (interaction mesh + IK); no dynamics or actuator limits, so the G1 "cannot perfectly execute" it. Expect self-collision/contact/torque infeasibility on floor-contact phases – a tracking reward alone will need relaxed/phase-gated terms.
- **Licences (conflicting layers):**
  - HF repo file LICENSE = BSD-3-Clause (Unitree, "2016-2024 HangZhou YuShu"); card metadata has no licence field.
  - Source mocap: Ubisoft LAFAN1, https://github.com/ubisoft/ubisoft-laforge-animation-dataset (license.txt) = **CC BY-NC-ND 4.0**. Text: may reproduce/Share the material for NonCommercial purposes only, and "produce and reproduce, but **not Share**, Adapted Material" (NonCommercial only). A retarget is plausibly Adapted Material, so Unitree's BSD-3 relicence is doubtful; BSD-3 cannot override ND/NC upstream.
  - Practical reading: private, non-commercial research training on it is allowed (portfolio/education counts as NC if nothing is monetised). **Do not redistribute the CSVs or any re-derived/re-timed motion files** (npz/pkl/converted) in the GitHub repo. A policy trained on it, plus metrics/videos, is generally not a copy of the data, but ND/NC ambiguity is real; videos of the reference playback itself (kinematic replay) are the riskiest – show only policy rollouts. Credit: Harvey et al. 2020 (LAFAN1, "Robust Motion In-betweening", SIGGRAPH) + Unitree retargeting; link dataset, don't bundle.
- Mapping effort: **low–moderate.** Reorder by joint name to the Isaac Lab G1 articulation order (Isaac Lab orders by USD tree, interleaved L/R – do not index by position), resample 30→50 Hz (interp; slerp the quat), hold the extra hand joints (37 − 29) at default, and set root from the CSV. Check the Isaac Lab G1 asset's default pose/axis signs vs URDF.

### 1b. Others
- AMASS/other retargets: no public, licensed G1 get-up retarget found. AMASS licence is non-commercial, no redistribution; own retarget would be needed (high effort). Not recommended.
- HoST and HumanUP release **no motion data** (both learn from scratch/RL; HumanUP stage II uses its own stage-I rollouts as a prior, not mocap).

## 2. Open implementations

| Repo | Sim / robot | Licence | Date | Notes |
|---|---|---|---|---|
| HoST – https://github.com/InternRobotics/HoST (ex OpenRobotLab; RSS'25 best-systems finalist; 644★) | Isaac Gym (legged_gym+rsl_rl, pytorch 1.10), **G1 23-DoF** (+H1, H1-2, Mini Pi) | **MIT** (c) 2025 OpenRobotLab | pushed 2025-06-17 | Ground/platform/wall/slope + prone/side. Most useful reward source. |
| HumanUP – https://github.com/RunpeiDong/HumanUP (RSS'25, 236★) | Isaac Gym, **G1** | **Apache-2.0** | pushed 2025-04-14 | Supine and prone, slopes; docs/install.md, simulation/README.md. |
| HoST-pytorch – https://github.com/lucidrains/HoST-pytorch | PyTorch re-implementation of the algorithm, no sim | MIT | pushed 2026-11-01 | Algorithm reference only. |
| g1-humanoid-rl – https://github.com/chaoticcuriosity-io/g1-humanoid-rl (5★, tiny) | **mjlab (Isaac-Lab-style API on MuJoCo-Warp)**, G1 | MIT | pushed 2026-11-05 | `Mjlab-Recovery-Flat-Unitree-G1`: from-scratch get-up, 4 lying reset poses (supine/prone/L/R side), no reference, 4 reward iterations. Closest to our backend stack; task source in recovery-task/, write-up docs/reports/14-building-get-up-from-scratch.md. |

No Isaac Lab (manager-based) or Newton get-up task found for G1 (searched GitHub for isaaclab/mjlab/newton get-up/standup/fall-recovery). So **nothing ports directly**; all need re-implementation of the MDP, reading only config/reward ideas (code licences MIT/Apache permit adapting with attribution, and notice retention; copying is fine in a public repo with attribution).
Mapping: HoST/HumanUP use 23-DoF G1 (no waist roll/pitch, no wrist pitch/yaw, no hands) and legged_gym URDF "keyframe" links (HoST adds head/torso keyframes and ankle points to its URDF). In Isaac Lab we have 29 + hand joints: lock or hold waist roll/pitch, wrist pitch/yaw and fingers at default (or strongly penalise), or keep them free and accept more exploration. Reward code must be rewritten against Isaac Lab body/sensor APIs (mapping effort moderate). Both papers sim-to-real tuned the collision model (ankle points rather than meshes) – matters on Newton too.

## 3. Reward / curriculum ideas in the literature
(HoST config g1_config_ground.py, README tips, HumanUP paper abstract/sections; g1-humanoid-rl ch.14)
1. **Staged task rewards by height** (HoST): stage 1 right body (rolling/sitting), stage 2 rise, stage 3 stand; stage boundaries ~35%/70% of robot height; head-height target ~75%, base height target 0.45→0.65 m. Orientation (torso up-vector) gaussian + head-height reward.
2. **Reward groups with a multi-critic**: groups task / regularisation / style / post-task, weights [2.5, 0.1, 1, 1]; separate critics per group (avoids small penalties being swamped).
3. **Assistive pulling force curriculum** (HoST): upward force on torso ~60% of robot weight (curriculum cfg force=100), applied while base height < ~0.7·height, decayed as success rises; removed to train unassisted.
4. **Style/posture penalties** (HoST): waist, hip yaw/roll, shoulder roll deviation (-10/-2.5), knee deviation, feet distance, ground-parallel feet; post-task rewards for zero feet/base vel, upper-body target joints, final stand pose.
5. **Regularisation for deployable motion**: action rate -0.01, smoothness -0.01, torque, joint power, dof acc/vel, strong dof-limit penalty (-100), action rescaler 0.25–0.3; "implicit speed bound" via dof-vel limit.
6. **Two-stage HumanUP curriculum**: stage I discovery with weak regularisation/simplified collision/canonical posture; stage II adds full collision mesh, random lying posture, strong smoothness/torque limits and DR. Hard-to-easy task + easy-to-hard sim2real.
7. **Posture randomisation**: supine, prone, side, random initial joints (HoST scale 0.9–1.1, offset ±0.1); DR: friction 0.1–1, motor strength 0.9–1.1, kp/kd 0.85–1.15, payload, delays up to 5 steps, terrain.
8. **Pitfalls (g1-humanoid-rl)**: no velocity-tracking reward (still robot scores max); gaussian height reward has no gradient at floor – use a monotonic height ramp; verify the reset really spawns on the ground; don't terminate on success; judge on video, not return.

## RECOMMENDATION
**No suitable reference for imitation; use fallback design Z** = HoST/HumanUP-style from-scratch RL (no mocap).
- The only data (Unitree LAFAN1 G1 `fallAndGetUp*`, 30 FPS, 29 DoF) sits under CC BY-NC-ND 4.0 upstream; it is not redistributable, is kinematic-only, and needs segmentation/retime. It is acceptable only as an *optional private, non-commercial* reference (never committed, never replayed in published videos; cite Harvey et al. 2020 + Unitree if used). Not worth the risk for a public portfolio repo.
- Fallback Z: Isaac-Velocity-Flat-G1 base, lying-pose reset distribution (supine/prone/sides), monotonic height-ramp + upright staged rewards, grouped multi-term regularisation (action rate, joint limits, torque), assistive-force then stabilisation curriculum, DR on friction/strength/gains. Adapt ideas with attribution: HoST (MIT, arXiv:2502.08378, Huang et al., RSS 2025), HumanUP (Apache-2.0, arXiv:2502.12152, He, Dong et al., RSS 2025), optionally g1-humanoid-rl (MIT) as an mjlab-style template.
- Lane B differs from Lane A by using HoST's pulling-force/stage curriculum + style reward groups (A = pure task reward).
