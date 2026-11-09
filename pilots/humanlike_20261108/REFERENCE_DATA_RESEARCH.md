# Human floor-getup reference data: what exists, rights, blockers (checked 2026-11-08)

Only public pages were read. Nothing was downloaded, no account or purchase was needed or made.

## Findings
| source | what it has | rights | status for this project |
|---|---|---|---|
| LAFAN1, Ubisoft La Forge, https://github.com/ubisoft/ubisoft-laforge-animation-dataset | human mocap (BVH): "Fall and get up" 6 sequences, "Push and fall" 2 sequences (repo README) | **CC BY-NC-ND 4.0** (repo `license.txt`; code MIT). NoDerivatives: a retargeted copy may count as a derivative, so do not commit or redistribute it; non-commercial research use only. Legal reading is not mine to settle. | Only found real human get-up mocap. Coverage by direction (back/front/side) **not verified**: needs viewing the clips. |
| Unitree retargeted LAFAN1, https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset (mirror; Unitree's own repo `unitreerobotics/LAFAN1_Retargeting_Dataset` returned 401/404 to my fetch) | CSV at 30 FPS: root (xyz + quat xyzw) + 29 joint angles, G1 order starting left_hip_pitch; also H1, H1_2 | Dataset card states the LAFAN1 license (CC BY-NC-ND 4.0) | Exact get-up file names in it **not verified** (page listing not readable). Mirror, not the publisher: prefer Unitree's own repo once readable. |
| openhe/g1-retargeted-motions, https://huggingface.co/datasets/openhe/g1-retargeted-motions | 40 LAFAN1-derived G1 files incl. "falls and get-ups" | "research purposes"; defers to LAFAN1 license | Third-party derivative; same restriction. |
| HumanUP, https://humanoid-getup.github.io/ , https://github.com/RunpeiDong/humanup (Apache-2.0) | RL discovery then tracking of the policy's **own** rollouts | Code Apache-2.0 | **No human mocap.** Not a data source. |
| HoST, https://github.com/InternRobotics/HoST | RL, no human reference | MIT per repo notes (not re-verified today) | No human reference. |
| CMU / AMASS / others | not checked today | n/a | Open item; AMASS needs account/registration (not done). |

## Compatibility with this simulator (measured from the trace joint list)
This env's G1 has 37 joints: 12 leg joints, **one** waist joint (`torso_joint`), per arm shoulder pitch/roll/yaw + elbow pitch + elbow roll, and 7 Dex3 finger joints per hand. The Unitree CSV is 29-DoF: it has waist roll and pitch and wrist pitch/yaw that this env lacks. A column copy is not valid; getting up uses waist pitch heavily, so the motion needs re-targeting (IK/optimisation) to this morphology, or the env needs the 29-DoF robot. Also: 30 FPS data vs this env's 50 Hz control (resample), different default pose.

## Trackers
- BeyondMimic `whole_body_tracking`, https://github.com/HybridRobotics/whole_body_tracking (MIT): Isaac Lab 2.1.0 / PhysX, G1 task `Tracking-Flat-G1-v0`, DeepMimic-style rewards. **Archived 2026-10-16**, points to mjlab. Needs a W&B registry account for motions (not allowed here) and a robot-description download from a Google bucket. Not drop-in for Isaac Lab 3.0 / Newton.
- mjlab tracking task, https://github.com/mujocolab/mjlab/tree/main/src/mjlab/tasks/tracking (active; MuJoCo-Warp, Isaac-Lab-style manager API): closest to this repo's MJWarp backend, but a different framework. Licence not re-checked today. Using it means a separate environment, not an edit to `g1_recovery`.
- Nothing in this repo tracks a reference: no loader, no tracking reward (`g1_recovery/` has HoST-style assist and a pose bank only).

## Blockers and decision for the parent
1. Data: LAFAN1 get-up clips are the only real human data found; licence blocks redistribution of retargeted files (NoDerivatives). Local research use only, files kept out of git.
2. Morphology: no waist roll/pitch in this env; needs re-targeting or a 29-DoF model.
3. Coverage: back/front/side not verified; the eval needs all four directions, so a few clips cannot cover HELDOUT. Realistic scope: one or two reference clips used as a style target near the start state, not a replacement for the policy.
4. Trainer: none in repo; BeyondMimic is archived and PhysX/2.1; mjlab is another framework.
5. Next validation step (lightweight, no training): the parent confirms Kaleb accepts LAFAN1's non-commercial/no-derivatives terms for local use; then download Unitree's official CSVs locally, plot the 6 clips (root height, hip roll/yaw, foot spread) to see directions covered and whether the human motion itself keeps feet near shoulder width. No imitation run starts before that and before parent confirmation.
