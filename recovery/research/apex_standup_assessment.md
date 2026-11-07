# APEX-humanoid `Isaac-StandUp-G1` / `standup.pt`: assessment (read-only, 2026-11-06)
Repo: https://github.com/apex-humanoid/APEX-humanoid (pushed 2026-10-02, 3 stars, BSD-3-Clause per LICENSE). Static analysis only: nothing was run, no GPU used,
files downloaded to /tmp/apex_dl (untrusted; checkpoint inspected by unzipping, USD parsed read-only with pxr).

## What it is
- A vendored fork of **Isaac Lab 1.3.0 / Isaac Sim 4.2 (PhysX)** with six G1 skills (walk, crawl, climb up/down, stand up, lie down). Task `Isaac-StandUp-G1`
  (`standup_env_cfg.py`), PPO run name `g1_standup_23dof` (OnPolicyRunnerLip/PPOLIP, 18000 iterations, 4096 envs, seed 42, symmetry augmentation).
- `checkpoints/standup.pt` (1,828,214 B, sha256 773fc75b... matches SHA256SUMS/manifest): **TorchScript** actor + observation normalizer (EmpiricalNormalization inside forward),
  MLP 558 -> 512 -> 256 -> 128 -> 23, ELU, no critic/optimizer (inference only; cannot resume training).

## Task facts that matter for comparison
| Item | APEX Isaac-StandUp-G1 | Ours (G1-Recovery, Lab 3.0.0 Newton/MJWarp) |
|---|---|---|
| Physics | PhysX (Isaac Sim 4.2), dt 0.005, decimation 4 (50 Hz) | Newton MJWarp, dt 0.005, decimation 4 (50 Hz) |
| Robot | custom `g1_29dof_modified_new_91.usd` (21 MB, in repo): 29 revolute DoF (3-DoF waist, 1-DoF elbow, 3-DoF wrist, **no fingers**), extra fixed mounts (d435, mid360, waist_support); 25 collision prims | Lab `G1_CFG` g1.usd: 37 joints (1-DoF torso_joint, 2-DoF elbow, 7+7 hand joints), full collision |
| Actuators | IdealPD, e.g. hip 100/4, knee 150/5, ankle 40/2, effort 88/139/50, vel limits | Lab G1_CFG implicit actuators (different gains/limits) |
| Actions | 23-d output, scale 0.25 about default pose (hips -0.1, knee 0.3, ankle -0.2, rest 0); the 6 wrist joints (indices 23-28) are not in the 23-d action (per symmetry code) | 37-d, scale 0.5 about default pose |
| Joint order | PhysX breadth-first: lhp, rhp, waist_yaw, lhr, rhr, waist_roll, lhy, rhy, waist_pitch, lk, rk, lsp, rsp, lap, rap, lsr, rsr, lar, rar, lsy, rsy, le, re, lwr, rwr, lwp, rwp, lwy, rwy (from their symmetry.py) | Lab 3 depth-first per leg, torso_joint, arms, fingers |
| Policy obs (558) | 6-step history of [base ang vel 3, projected gravity 3, joint_pos-default 29, joint_vel 29, last PROCESSED action (target positions) 29] = 93 x 6; noise on; no velocity command, no base height | 123-d single step incl. lin vel, command, raw last action |
| Episode | 7 s, pushes every 1-3 s (+-0.5 m/s, rot too), domain randomisation on in play | 10 s, no pushes |
| **Start poses** | root = init rot (0.7071,0,0.7071,0) = 90 deg pitch **= prone (face-down)**, z 0.4 m drop; reset roll +-pi applied in the BODY frame (quat_mul(init, delta)) = random HEADING only; yaw +-15 deg; joints = fixed `lying_joint` posture (hip pitch -1.5, knee 1.5, ankle -0.23, shoulder pitch -1.8, roll +-0.5) +-0.1 rad | held-out: 256 lying poses, 4 fall directions (supine, prone, 2 sides), random limbs +-0.4 rad |
| Success | no explicit metric; rewards: shoulder height 1.07 m, feet orientation, penalties for kneeling/leg twisted/feet off ground terminations | pelvis >= 0.60 m & torso tilt <= 26 deg held 2 s |

So the policy was trained from ONE canonical prone, knees-tucked, arms-up posture (the end pose of their `Isaac-LyDown-G1` skill) with random heading. It was not trained for supine, side or arbitrary falls.

## Licence / attribution
- LICENSE is Isaac Lab's BSD-3-Clause text with "Copyright (c) 2022-2024, The Isaac Lab Project Developers"; CONTRIBUTORS.md is Isaac Lab's list. No named APEX authors, no CITATION file, no paper link. docs/licenses lists Unitree BSD-3 for assets.
- `standup.pt` and the modified USD carry no separate licence statement; the USD is a modified Unitree model with unknown provenance for the modifications. Fine to evaluate privately and cite the repo; do not redistribute the checkpoint/USD in our public repo without clarification.

## Can it be evaluated here?
- **As-is on our Newton env: no.** Different robot variant/joint set, joint order, obs layout (history 6, no command/lin vel), action semantics, actuators. Feeding our 123-d obs would be meaningless.
- **Option 1, native (own stack, PhysX):** install Isaac Sim 4.2 + Python 3.10 + their vendored Lab 1.3.0 in a separate environment and run their `play.py --torchscript` on their task to reproduce what it does (a baseline for their start distribution only). Needs a new env install (several GB; Isaac Sim 4.2 binary not present here: we have 6.0.1/6.1/6.2 apps).
- **Option 2, port as a separate comparison (Lab 3.0.0, Newton or PhysX):** new env cfg on `G1_29DOF_CFG` (29-DoF, no hands) + a 558-d observation builder with a 6-frame history buffer (ang vel, gravity, joint_pos - default, joint_vel, processed targets) in THEIR joint order, an index remap to Lab 3's order, 23-d actions (scale 0.25, offset default; wrists held at default/0), their IdealPD gains/effort/velocity limits (Lab 3 Newton actuator support must be checked), their prone start (pitch 90 deg, z 0.4, lying_joint posture, yaw +-15 deg, random heading) as one test set, then OUR success metric on both their start set and our held-out set. Expected: works only near its training distribution; transfer to Newton is a real risk (our own PhysX-trained walking policy scored 0/256 on Newton earlier).
- **Unresolved by static analysis:** the repo's config uses `joint_names=[".*"]` (29 action dims) but the checkpoint outputs 23 and docs say 23; the symmetry code says "23-DoF action (excludes wrist joints)". The exact joint-to-output mapping and how wrists are driven must be confirmed by running their env (Option 1) before trusting a port.

## Recommendation
Treat it as an **external reference baseline on its own start distribution (prone, tucked)**, via Option 2 only after confirming the mapping with Option 1 (or by diffing their action term at runtime). Do not claim it supports arbitrary falls. Reuse ideas for Lane B: staged shoulder-height reward, strong penalties on kneeling/feet-off-ground, `lying_joint`/`standup_joint` keyframes, 6-step observation history, symmetry augmentation, 18k-iteration budget (vs our ~600-1000).
