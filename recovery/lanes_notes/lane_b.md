# Lane B notes (G1 get-up, HoST/HumanUP-style, no mocap)

No licensed/usable G1 get-up reference exists (see research/lane_b_references.md), so Lane B is from-scratch RL with a different
reward + curriculum design than Lane A. Credits: HoST (Huang et al., arXiv:2502.08378, MIT), HumanUP (Dong et al., arXiv:2502.12152, Apache-2.0).
Terms re-implemented, no code copied.

## Reward (g1_recovery/lane_b.py)
Pelvis-height stages (boundaries 0.35/0.70 x 0.74 m = 0.26/0.52 m):
- task: `right_body` (torso up, fades out between 0.26-0.52 m) w1; `rise` (monotonic pelvis ramp x torso-upright factor) w3; `stand` (smoothstep above 0.52 m x sharp torso-upright gaussian) w2.
- post-task (only while the success criterion holds): `still` w2, `stand_pose` (default-ish pose, no hands/wrist) w1.5.
- regularisation: action rate -0.01, dof acc -2.5e-7, joint-limit -2.0, torque -5e-7 (hand joints excluded; their default lies outside soft limits).
- style: hip yaw/roll, shoulder roll, waist deviation (L1) -0.1 each.
Lane A: 4 task terms (upright, height, stand bonus, default pose) + 4 regularisers (limits/accs/torques over all joints incl. hands).

## Assist
World-frame +z force on `torso_link` via `Articulation.permanent_wrench_composer.set_forces_and_torques_index(is_global=True)`, written each control step by
a never-terminating "termination" hook (`assist_hook`); Newton applies it in `write_data_to_sim` every physics sub-step. Magnitude =
0.6 x scale x robot weight, gated by pelvis height (full below 0.458 m, taper to 0 at 0.518 m = 0.7 x 0.74).
`env._assist_scale` defaults to 0 and is only set by the training curriculum term (`assist_curriculum`, runs on reset). Eval/record scripts never
invoke curriculum -> exactly zero force (hook does not even call the wrench API). Curriculum: scale starts 1.0; success EMA over finished episodes;
while EMA > 0.30 scale -= 0.08 per "generation" (num_envs resets); never increases. Logged as Curriculum/assist, Curriculum/succ_ema.

## Matched with A / differences
Matched: robot cfg, physics, obs, start-pose distribution (same reset_fallen, same `promote_demote` per-env levels, same alpha/noise), success def, PPO cfg.
Differences: reward shape (above), the assist force during training only, and style/regularisation excluding hand joints.
Caveat: success EMA during training is measured with assist active; the assist decays toward 0 only if success persists, so the final policy trains unassisted for the tail of a long run (scale reaches 0 after ~13 generations of EMA>0.3).
