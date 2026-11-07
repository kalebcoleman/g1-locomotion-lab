# HoST and HumanUP vs our G1 recovery implementation (code-level read, 2026-11-06; read-only, nothing run)
Sources: https://github.com/InternRobotics/HoST (MIT, pushed 2025-06-17; legged_gym/g1/g1_config_ground*.py, envs/base/host_ground.py) and
https://github.com/RunpeiDong/HumanUP (Apache-2.0, pushed 2025-04-14; simulation/legged_gym/.../g1waist/g1waist_up_config.py, README). Both Isaac Gym + legged_gym/rsl_rl (PhysX), custom G1 23-DoF URDF (no wrists/hands). Licences permit adapting ideas with attribution; we re-implement, not copy.

## What they actually do
| | HoST (ground, supine; `_prone` variant) | HumanUP (stage I discovery, then stage II tracking) | Ours (v1-v6) |
|---|---|---|---|
| Training budget | **12,000 iterations**, 4096 envs x 24 steps (~1.2 B transitions) | **50,001 iterations**, 4096 envs, regularisation/standing-scale curricula run to 50k-100k iterations | 600-1,000 iterations (25-59 M transitions) |
| Start | root z 0.5, rot pitch -90 (supine) or +90 (prone), default joints +-10 % scale, +-0.1 offset; **one posture family per policy** (joint supine+prone training listed as open problem; hard hip constraints needed for prone) | root z 0.8 posture bank (`facingup_poses.npy`, `facingdown_poses.npy`) for random lying postures at stage II; stage I canonical | 256 held-out lying poses, 4 directions, random limbs +-0.4 rad (much broader) |
| Settling | **first 30 control steps unactuated** (policy output zeroed, 0.6 s) so the body settles before acting; pull force also gated by this | n/a | policy acts from step 0 |
| Actions | PD target = rescale x action + current dof pos? (HoST: `action_scale 1`, per-env `action_rescale` curriculum shrinking 0.02/episode toward 0.25); 23 DoF | scale 0.5 about default, 23 DoF | scale 0.5 about default, 37 DoF (fingers included) |
| Observation | 76-d one step x **6-step history** (actor), privileged critic | proprio 3+2+3*23, history 10 + privileged latent (RMA-style history encoder) | single 123-d step (matches walking policy) |
| Assist | **vertical pull force on torso ~60 % weight, applied only when the torso is already fairly upright** (`projected_gravity_z < -0.8`) and after the unactuated period; **per-env** force decays 20 N per episode (100 -> 0) when that env's mean head height beats 0.9 (so removed within ~5 successful episodes) | none (hard-to-easy via stage I simplified collision, then tracking) | global scalar assist, applied when pelvis < 0.46 m regardless of torso angle; decays only on a global success EMA (never reached) |
| Rewards | 4 groups (task, regularisation, style, post-task) with **multi-critic** (one value head per group) and group weights [2.5, 0.1, 1, 1]; task = torso orientation x head-height tolerance; staged base-height targets 0.45 -> 0.65; strong style terms (waist/hip yaw+roll -10, shoulder roll -2.5, feet distance -10, shank orientation +10, ground-parallel feet +20), dof limit -100; **no termination on contact** | base_height_exp 5, head_height_exp 5, delta_base_height 1, stand_on_feet 2.5, body_up_exp, feet_height/feet_distance, symmetry penalties; **termination -500** (on velocity/height), regularisation scale curriculum 0.8 -> 2.0 | single summed reward: upright 1, height ramp 2, stand bonus 5, pose 1 + small penalties (A); staged B variant similar |
| Domain randomisation | motor strength 0.9-1.1, kp/kd 0.85-1.15, friction 0.1-1, payload -2..5 kg, COM +-3 cm, link mass 0.8-1.2, action delay up to 5 steps | similar + motor-delay FFT | startup friction + torso mass only |
| PPO | init std 0.8, entropy 0.01, MLP 512-256-128 actor / 512-256 critic, value-smoothness loss | ActorCriticRMA / PPORMA | Lab's G1 PPO (256-128-128, init std 1.0 -> 0.94 after warm start) |
| Outcome claims | real-robot standing up from diverse postures incl. slopes/walls | real G1 get-up from supine and prone | 0/256 held-out success so far; v1 reaches sitting only |

## Gaps in our implementation, ranked by expected impact
1. **Training budget is 1-2 orders of magnitude too small** (25-59 M vs ~1.2 B / ~4.9 B transitions). Lying-to-standing is a long-horizon exploration problem; our from-scratch run did discover lying->sitting, which is consistent with "needs more steps".
2. **Assist gating**: HoST applies the pull only after the torso is upright (not while lying), per-env, with decay tied to each env's own success. Ours pulls the pelvis up while lying and decays on a global rate that never triggered, so B learned to rely on it. Fix: gate on `projected_gravity_z < -0.8`, per-env force level, decay when that env's recent pelvis/head height exceeds target.
3. **Unactuated settling (0.6 s)** and **action-scale curriculum** (large scale decaying to a floor) are cheap and absent.
4. **Posture difficulty**: both papers train from ONE posture family with near-default joints; we start at the held-out difficulty (random limbs +-0.4 rad, 4 directions, mixed). Train per family (supine, prone) first, then mix; add the +-0.4 rad noise last.
5. **Observation history** (6 steps in HoST, 10 in HumanUP) is missing; our policy is memoryless. Chaining to the walking policy still works with separate observation groups.
6. **Reward structure**: multi-critic by group (needs a modified PPO; rsl_rl 5.5.1 here has one critic) vs our summed single reward; no style/foot constraints; HoST's head-height tolerance (relative to feet) vs our pelvis ramp.
7. **Self-reference (HumanUP stage II idea, no licensing issue)**: log our own best rollouts (e.g. A v1 lying->sit) and train a tracking policy on that trajectory; this is the legitimate "reference-guided" lane since no licensed external get-up motion exists (see lane_b_references.md).
8. Domain randomisation (kp/kd, motor strength, delay) is irrelevant to feasibility but required for any robustness claim.

## What I would change next (not started; v6 still running)
- v7: same env, but (a) per-env, torso-upright-gated assist with per-env success decay, (b) 30-step unactuated start, (c) action-scale curriculum 1.0 -> 0.5, (d) train on ONE posture family first (supine), canonical joints, (e) 6-step history, (f) 3,000+ iterations at 4096 envs (~300 M transitions, ~2 h with two jobs).
- Optional lane C: HumanUP-style self-tracking from our own rollouts.
- Keep Lane A = unassisted task reward with the same budget/posture schedule so the A/B comparison stays about the assist and reward design.
