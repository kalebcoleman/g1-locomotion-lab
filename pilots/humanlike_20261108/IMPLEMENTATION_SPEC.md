# M1 implementation spec and status (smallest useful set, in order)

**Status on this branch:** items 1 (except the thigh abduction angle, deferred: sign convention unverified) and 2 are implemented and validated on the frozen baseline (see `HANDOFF.md`, "M1 validation"). Items 3 to 6 are not implemented.

Context and measurements: `HANDOFF.md`. No root cause is claimed; items below make the cause measurable and the penalty configurable. Paths relative to `recovery/`.

## 1. Eval metrics first (`scripts/eval_recovery.py`, `scripts/diag_trace.py`, `g1_recovery/metrics.py`)
Today `eval_recovery.py` keeps per-step foot distance only as `_A["stance"]` and reports its mean over the counted hold; `diag_trace.py` saves `stance` per step and `q_end` only.
Add, per robot and aggregated (mean, median, p90, max) and by fall direction:
- `foot_spread_peak_m`: max over the episode of the xy distance between the two `ankle_roll` links (as `stance`), over steps with pelvis height > 0.12 m; also report the value with the existing 0.40 m gate.
- `foot_spread_peak_time_s`, `foot_spread_final_m` (mean of last 1.0 s), `frac_steps_spread_gt_0.45m` (gate: pelvis > 0.40 m).
- Hip: per-step `hip_roll`, `hip_yaw` joint positions (4 joints); report peak |value| and final |value| (rad). The thigh abduction angle is deferred (its sign convention is unverified).
- Smoothness over the rising window (reset to start of the counted hold): mean |a_t - a_{t-1}| (action rate, from the action manager), RMS and p99 of joint acceleration (finite difference of `joint_vel` at 50 Hz, body joints), p99 jerk. State units.
Acceptance: on `ckpt` baseline (`sha256 665a89f58b3a...`) and `sets/heldout_v2.npz`, peak spread after the 0.40 m gate has mean 1.32 m and final spread 0.427 m (interim numbers from `pilots/humanlike_20261108/analyze_trace.py`, tolerance 0.01 m); unit tests in `tests/test_metrics.py` for each metric on synthetic traces; existing 12 tests still pass; v3 counts unchanged for the same checkpoint.

## 2. Configurable early-rise foot-spread penalty (`g1_recovery/lane_a.py`)
Current: `stance_excess(env, free=0.45, full=0.90, pelvis_h=0.40)` (line ~364), used as `RewTerm(func=stance_excess, weight=-5.0)` in `RewardsR9h` with empty `params`. Hydra rejects `+env.rewards.stance.params.free=...` ("Key not found under namespace"), so the gate and band cannot be changed from the command line.
Required:
- Declare the params explicitly in the term (`params={"free": 0.45, "full": 0.90, "pelvis_h": 0.40}`) so `env.rewards.stance.params.pelvis_h=0.15` is accepted. Defaults must keep R9h/R9hc/R9hg behaviour bit-identical.
- Allow `pelvis_h` low enough to cover the rise (measured: spread peaks at t≈0.35 to 0.5 s with the pelvis 0.3 to 0.45 m; the lying start is 0.1 to 0.2 m, so `free` around 0.30 m does not penalise the start).
Acceptance: 3-iteration smoke (256 envs, as in the pilot's `train_pilot.sh`) accepts `env.rewards.stance.params.free=0.30 env.rewards.stance.params.pelvis_h=0.15`; `Episode_Reward/stance` is non-zero during the rise; with defaults, an R9hg eval of the frozen checkpoint gives the same v3 count as before (131 and 122 observed today, so compare within that range).

## 3. Hip-splay term (only if item 1 shows where the spread comes from)
Add an optional reward `sum over hip roll/yaw joints of relu(|q - q_default| - deadzone)^2`, same gating params as item 2, deadzone and weight configurable, default weight 0 (no behaviour change). Justification so far is weak: final hip roll/yaw are small (mean |roll| 0.14 rad, |yaw| 0.25 rad at the end), the early-rise values are unmeasured. Do not add before item 1.
Acceptance: weight 0 reproduces baseline eval; smoke run logs the term.

## 4. Pilot rerun gate (parent decides; not authorised yet)
One bounded run only after 1 and 2: same warm start, same seed 21, 100 iterations, DEV for tuning and the early-stop check, HELDOUT once at the pre-declared checkpoints. Fix the driver glob that failed in the pilot (run folder is `<timestamp>_<run_name>`). Success rule stated up front: v3 within baseline range (122 to 131) AND peak spread after rise below 1.0 m AND soft-limit rule fails not above baseline.

## 5. Reference tracking (blocked on data; see `REFERENCE_DATA_RESEARCH.md`)
Minimal scope once a clip is approved and re-targeted to this 37-joint morphology (one waist joint):
- CSV loader (frame, root xyz + quat xyzw, joints; 30 FPS) with resample to 50 Hz and a joint-name map to this env; fingers fixed at default.
- Reward: DeepMimic-style tracking terms (joint position, root height/orientation, optionally foot position) with a phase variable, added to the existing getup rewards at a configurable weight; weight 0 = current behaviour.
- Start states from the clip's first frames in addition to the existing pose bank.
- Metrics: tracking error per direction next to v3.
Acceptance: loader unit test (frame count, order, quaternion convention, 30 to 50 Hz resample); a replay script that sets the reference pose and renders one clip (visual check); a 3-iteration smoke with the tracking term on. No training run until the parent confirms the bounded run.

## 6. Isaac MCP gap
`isaaclab__envs_list` / `isaaclab__train` cannot see `g1_recovery` tasks (needs `PYTHONPATH=<recovery>` and `--external_callback=g1_recovery.register.register`). Ask: a way to pass env vars or a Python path to MCP jobs, or a registered entry point.
