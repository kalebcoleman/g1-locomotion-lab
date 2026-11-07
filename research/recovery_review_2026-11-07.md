# G1 recovery: independent read-only review (2026-11-07)

Scope: the live recovery project (`projects/g1-locomotion-lab-integrated/recovery/`, private, local), its R6c/R7 runs, reward code, evaluation protocol v2.1 and job logs. Nothing was launched, stopped or changed; the TEST pose set was not read. V = verified in code or logs, H = hypothesis.

## 1. State at review time

Under protocol v2.1 there is still no valid unassisted recovery. The R6c control converged to a static forward fold; the three R7 reward variants were near iteration 260 of 1000 when read. Values below are rsl_rl per-second reward-term averages parsed from the MCP job stdout logs.

| run | iteration | height term | upright term | stand term | lying max_h | action std |
|---|---|---|---|---|---|---|
| R6c control (scratch) | 710 / 1000 | 8.17 | 0.001 | 0.000 | 0.66 | 0.90 |
| R7u orientation gradient | 257 | 7.29 | 0.28 | 0.000 | 0.61 | 0.91 |
| R7h imu_link height | 256 | 6.45 | 0.56 | 0.000 | 0.58 | 0.94 |
| R7m height gated by orientation | 256 | 4.58 | 1.37 | 0.000 | 0.52 | 0.97 |
| R7u warm from R6c fold | 605 / 1350 | 7.92 | 0.22 | 0.000 | 0.64 | 0.92 |

V: R6c's height term implies a mean pelvis height near 0.62 m with the torso inverted. `runs/probeH_V.log`: pelvis 0.52 m, imu_link 0.38 m, pelvis up-vector -0.97 in the fold. `results/chk300_R6cS_dev_off.json` (it 350, DEV): 256/256 `rose_not_upright`, zero supported stands, soft position-limit violations on 40% of steps, peak pelvis vz mean 1.24 m/s. R7m was the only variant whose `pose` term and `lying_ever_supported` had moved off zero at that point.

## 2. Why the policy stalls (ranked)

1. **Pelvis-height ramp is maximised by the fold (V shape, H mechanism).** `height_ramp_floor` (weight 10) rewards pelvis height alone; `upright` is clamped at zero below horizontal; the stand bonus (weight 5) needs pelvis >= 0.60 m and up >= 0.90 at once. A four-point stance on straight arms and legs reaches near-standing pelvis height with no balance risk. Lifting the hands means a squat that temporarily lowers the pelvis. g1-humanoid-rl ch. 14 hit the same squat valley with the same ramp.
2. **Static-favouring penalties (V weights, H effect).** Body position-limit -10, velocity-limit -10, launch -5, flight -5. R6c drives all of them to ~0 by being motionless; a fast tuck near limits is the transition.
3. **Exploration noise never shrinks (V).** Action std 0.90 to 0.97 in every scratch run (about 0.45 rad target noise at scale 0.5). The walking run also ends at 0.94, but it trains with a -200 fall termination that selects for noise-robust balance; recovery has only a time-out, so a stance that never falls (fold) beats a stance that falls under noise. H.
4. **Mixed posture families from step zero, no settle (V setup).** Reset drops from 0.20 to 0.30 m with +-0.3 to 0.4 rad limb noise, four directions, policy acting mid-drop. HoST trains one family per policy with 30 unactuated steps and reports supine+prone interference. H: direction mixing dilutes the gradient.
5. **Assist gate can never open from the fold (V).** `assist.py` fires only when up >= 0.8 and pelvis < 0.62 m; the fold has up -0.97. Explains the V12 "no force even when enabled" entry.
6. **Budget.** One R-run is about 393 M transitions (1000 x 16384 x 24). HoST about 2.5 B. g1-humanoid-rl reached a stand at about 123 M on a much easier start distribution.

## 3. Reward exploits and evaluation gaps

- V: the stand bonus in every baked R recipe uses the legacy mask (`G1R_STRICT` unset in `common.standing_mask`). Training rewards an unsupported stand; evaluation requires both feet loaded. Shared by all arms; fix in the next recipe.
- V: Isaac Lab `G1_CFG` gives the legs and the whole arm group, including the 14 finger joints, a 300 Nm effort limit, and `NewtonArticulationCfg(self_collision_enabled=False)`. Real G1 knee about 139 Nm, arm joints about 25 Nm (datasheet values as used in HoST and unitree_rl_gym configs; verify). No criterion checks torque against real limits or limb interpenetration. Proposed Amendment 3: torque within real limits on >= 97% of steps and no self-penetration, frozen now while no result can be favoured.
- V: validity fractions are per 10 s episode. A 2 s rise plus 8 s hold can violate position limits for a full second and pass the 10% rule. Also report fractions over the rising window (reset to start of hold).
- V: `torso_link` and `head_link` report the pelvis origin in this USD (`runs/probeH_stand.log`), so the assist force on `torso_link` is a pelvis-origin lift with no righting torque. Height now uses imu_link.
- V: DEV is the old held-out set and is used for pruning; HELDOUT is clean; TEST untouched. Evaluation runs with observation noise and is not bit-repeatable (3 to 8 of 256 spread); the protocol's three evaluations per checkpoint covers this.

## 4. Do the R6c/R7 comparisons isolate their changes?

Mostly yes for R7u, R7h, R7m versus R6c: same seed 3, 16384 envs, PPO config, lying-only mix, scratch init, one reward term changed each. Caveats: one seed each; each change also alters reward magnitude relative to the fixed penalties (R7m caps height unless upright, R7h changes the ramp slope); R6c was checked at 350 while the rule says 300. The warm R7u resumes actor and critic from the R6c fold with the counter continuing to 1350, so its critic was trained on a different reward and it carries 350 extra iterations: a separate labelled arm, not a matched comparison. R6c versus R6b differs in init, seed and start mix, so "lying-only starts" is not isolated.

## 5. Next experiments

Each is a single change against the R7 winner (R7*), same seed, 16384 envs, 1000 iterations, run beside an R7* control replicate. DEV at 300 with video; HELDOUT with three evaluations at the end; TEST untouched.

**E1. Feet-load-share reward with a gated prop penalty (highest value).** Hypothesis: the fold persists because hands, head and shins carry load at no cost. Change: `feet_share = both_feet_in_contact * sum|F_feet| / (sum|F_all| + 1)` from the `contact_forces` sensor, weight 3; `prop = 1[pelvis > 0.40 m] * min(sum|F_hands,elbows,knees,head| / 100 N, 1)`, weight -2. Measure `lying_ever_supported`, `lying_strict_hold`, `crouch_on_feet` and `high_on_feet` posture flags, mean feet share, hand force, v2.1 by direction. Failure modes: one-foot or toe stances (keep the both-feet gate), pike on the feet (R7m gating counters it), refusal to use hands from supine (lower the gate or weight).

**E2. One posture family at a time with an unactuated settle.** Hypothesis: mixing four directions and acting mid-drop spreads the gradient thin. Change: supine-only and prone-only runs (restrict `phi` in `reset_fallen`), zero actions for the first 30 steps after reset. Measure per-direction v2.1 on DEV and HELDOUT, time to first supported stand, cross-direction transfer. A family policy plus a fixed switch on projected gravity is acceptable under the protocol.

**E3. Exploration schedule (after E1/E2 produce supported stands).** Hypothesis: the static optimum is partly a noise-robustness artefact. Change: warm from R7* at 300 (actor and critic, fresh optimiser, counter reset) with `init_std` 0.5 and `entropy_coef` 0.0 in `RecPPOCfgR`, or an action-scale curriculum 0.5 toward 0.25 as in HoST. Measure std trajectory, `Episode_Reward/stand`, supported time, v2.1. Failure: exploration collapse (hence the warm start).

Replicate any winner on two more seeds and three evaluations per checkpoint before any robustness claim.

## 6. Open-source references and what transfers

See `REFERENCES.md` for links and licences. Summary:

| source | licence | stack | what transfers |
|---|---|---|---|
| HoST (RSS 2025) | MIT | Isaac Gym, 23-DoF G1 | orientation x head-height task term, staged heights, feet-parallel and shank-orientation style terms, 30-step settle, action-rescale curriculum, multi-critic; re-implement as Isaac Lab terms |
| HumanUP | Apache-2.0 | Isaac Gym, G1 | `stand_on_feet`, head-height, terminate-on-velocity, two-stage collision curriculum |
| g1-humanoid-rl ch. 14 | MIT | mjlab / MuJoCo-Warp, G1 | same squat optimum; fixed with height 10, action-rate -0.03, no success termination; easier reset and no plausibility criteria, so not evidence for our task |
| mjlab | Apache-2.0 | MuJoCo-Warp, Isaac-Lab-style API | second simulator with Unitree's real actuator limits for a cross-check |
| unitree_rl_lab | BSD-3 | Isaac Lab, G1-29dof | official actuator parameters to replace the 300 Nm limits for a realism check |
| BeyondMimic / whole_body_tracking | MIT (retired into mjlab) | Isaac Lab 2.1, G1 | only with a licensed get-up clip; LAFAN1 retargets stay CC BY-NC-ND |
| FRASA | MIT | MuJoCo, 20-DoF kid-size | settle-before-act, velocity-mode actions, upside-down termination; ideas only |
| FIRM (arXiv 2511.07407) | no code found | G1 23-DoF, sim and real | success = base > 0.7 m sustained; unifies fall and recovery; reference only |
| APEX-humanoid | BSD-3 | PhysX Lab 1.3 | already assessed in `apex_standup_assessment.md`; prone-only, not portable |

## 7. Recommendation given to the training agent

1. Keep R6c and the R7 runs running; at iteration 300 evaluate all three scratch R7 variants on DEV with video; treat the warm R7u as a separate labelled arm.
2. Pick R7* as the new control; launch E1 and E2 as single changes against it; add E3 only once a supported stand appears.
3. Freeze Amendment 3 (torque versus real limits, self-penetration) and the rising-window validity report now, and re-evaluate existing checkpoints under it.
4. Make the stand bonus use the supported mask in the next recipe; record the reward-magnitude side effect of each R7 variant in the experiment log.
5. Replicate any winner on two more seeds and three evaluations per checkpoint on HELDOUT before touching TEST.

## Files inspected

`projects/g1-locomotion-lab-integrated/{STATUS.md,EXPERIMENT_LOG.md,EVAL_PROTOCOL.md}`; `recovery/g1_recovery/{lane_a.py,common.py,env_cfg.py,assist.py,metrics.py,agents.py,register.py}`; `recovery/scripts/{eval_recovery.py,train_recovery.py,mon.py,evalv2.sh}`; `recovery/{REPORT.md,CRITERIA.md}`; `recovery/research/*.md`; `results/chk300_R6cS_dev_off.json`, `results/chk300_R4nS_dev_off.json`; `runs/probeH_V.log`, `runs/probeH_stand.log`; `media/20261106_224419_chk300_R6cS_t1_3_6_9.png`; the five MCP job logs under `~/.local/share/isaac-mcp/jobs/`; Isaac Lab 3.0.0 `isaaclab_assets/robots/unitree.py`, `velocity_env_cfg.py`, `g1/flat_env_cfg.py`, `g1/agents/rsl_rl_ppo_cfg.py`, `mdp/rewards.py`, `isaaclab_rl/rsl_rl/rl_cfg.py`; rsl_rl `modules/distribution.py`.
