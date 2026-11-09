# Success criteria (frozen 2026-11-06 19:55 MST, BEFORE evaluating v9 and later runs). Local, reviewable; nothing published.
Everything is evaluated in Newton/MJWarp, Lab 3.0.0, full-collision G1, 50 Hz, on matched held-out poses (`heldout_poses.npz`: 256 lying poses, 4 fall directions x 64, limbs +-0.4 rad, seed 20261106, never used in training).

## Stage 1: PLAUSIBLE UNASSISTED RECOVERY (primary metric; `eval_recovery.py` -> `plausibility`)
All must hold in one uncut 10 s episode starting from the held-out pose, no reset or teleport after t=0:
1. Supported stand: pelvis >= 0.60 m, torso tilt <= 26 deg (up-vector.z >= 0.90), BOTH feet contact force > 10 N and < 0.15 m above floor, pelvis speed < 0.6 m/s, continuously for >= 2.0 s.
2. Valid joint velocities: any body joint above its soft velocity limit on <= 2% of the episode's steps.
3. No launch: peak pelvis upward speed <= 2.0 m/s.
4. No assistance: no external force (assist hook inactive or level 0; verified by the eval's `assist` block).
Also reported (not part of success): joint-position-limit violation fraction, actuator effort (torque/limit), contact forces on feet/hands/trunk, recovery time (start of the successful hold), longest/total supported stand time, failure cause histogram, per-direction rates.
Legacy criterion (pelvis+upright only) and the strict criterion without plausibility are always reported alongside, never as the headline.

## Stage 2: STAND -> WALK (chain evaluation, `eval_chain.py`)
After a Stage-1-style supported stand held for the hand-over rule, the existing walking policy (published run iteration 1499, `physics=newton_mjwarp`) takes over with command vx=0.8 m/s (optionally yaw 0.8 rad/s). Chain success = Stage-1 plausible recovery AND no fall after hand-over (pelvis < 0.45 m or tilt > 60 deg) AND >= 3.0 m net displacement along the commanded heading within 10 s of walking AND mean forward speed >= 0.5 m/s. Hand-over rule must be stated per result (default: 100 consecutive supported steps = 2 s).

## Stage 3 (stretch): CONTINUOUS WALK -> DISTURBANCE -> FALL -> RECOVER -> WALK/TURN
One uncut episode, no stitching of separate runs, no reset/teleport/hidden support/gravity or contact changes. Disturbance: an impulsive base velocity push (applied by the task's own push mechanism, magnitude and direction logged) that causes a fall (pelvis < 0.45 m or torso tilt > 60 deg). Then Stage 1 plausibility from the moment of the fall (ground contact), a supported stand, and resumed commanded walking (>= 3 m) plus a turn command (>= 90 deg net yaw within 8 s). A switching rule between policies must be a fixed, stated function of observable state. Report attempts/successes per fall direction, per held-out disturbance (magnitude/direction set fixed in advance), per seed.

## Evidence thresholds for any 'robust' claim
>= 3 training seeds each evaluated on all 256 held-out poses; report mean +- range; evaluation itself is not bit-repeatable (observed 3-8 of 256 variation for one checkpoint), so each checkpoint is evaluated >= 3 times. A single lucky rollout is never evidence. Complete uncut video of a success must be accompanied by the aggregate table. Training resets and temporary assistance are allowed only when labelled; assisted results are intermediate milestones.

## AMENDMENT 1 (2026-11-06 20:05 MST, POST-HOC, after seeing A8c/A8c_s3 results; labelled, v1 numbers are still reported)
The frozen v1 plausibility thresholds (<=2% joint-velocity-limit-violating steps, peak pelvis upward speed <=2.0 m/s) were too lenient: under v1, 51/256 (seed 3) and 10/256 (seed 2) of the A8c launch-style episodes pass although the policy launches itself. Added free-flight fraction (no body in contact with anything; contact-sensor net force summed over all bodies < 1 N): v1.1 plausible = v1 plausible AND flight <= 2% of steps. Eval JSON reports both (`plausible_successes`, `plausible_v1p1_successes`). v1.1 is only applied to runs evaluated after this amendment; earlier runs are re-evaluated with it where checkpoints exist.
