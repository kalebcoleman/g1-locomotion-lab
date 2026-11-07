# v7 pilot: gated per-env assist (treatment A7) vs A6 (control)
Single changed factor: A7 = `G1-Recovery-A-Assist-v0` with `G1R_ASSIST_TRAIN=1` (g1_recovery/assist.py); A6 = `G1-Recovery-A-v0`.
Everything else identical: init actor = Lane A v1 model_999, `G1R_START_LEVEL=1`, 4096 envs, 600 iterations, fixed lr 3e-4, seed 1, same rewards/curriculum/obs/bank, same eval.
Assist (re-implemented from HoST's idea, MIT, arXiv:2502.08378): +z force on torso_link = 0.6 x level x robot weight (310 N -> 186 N max),
applied only while root upright (projected gravity z < -0.8, as HoST) AND pelvis < 0.62 m (our addition: a standing robot is never held up);
per-env level 1.0 -> decays 0.2 each time that env's finished episode reached the standing criterion at least once (HoST: -20 N per episode where mean head height > threshold); never rises.
Not copied from HoST: unactuated first 30 steps (absent here), action-rescale curriculum (absent), force on a second virtual torso link (absent), head-height decay criterion (we use the standing criterion).
Eval: always report assist OFF (G1R_EVAL_ASSIST=0) and ON (1.0) with applied force statistics in the JSON ('assist' block). Assisted standing = intermediate milestone only.
