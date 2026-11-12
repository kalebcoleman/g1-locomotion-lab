<div align="center">

# G1 Locomotion Lab

**Teaching a simulated Unitree G1 humanoid to walk, survive pushes, and get back up after a fall, with reinforcement learning in Isaac Lab.**

[![Tests](https://github.com/kalebcoleman/g1-locomotion-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/kalebcoleman/g1-locomotion-lab/actions/workflows/tests.yml)
![Isaac Lab 3.0.0](https://img.shields.io/badge/Isaac%20Lab-3.0.0-76B900)
![Unitree G1](https://img.shields.io/badge/robot-Unitree%20G1-555)
![Simulation only](https://img.shields.io/badge/scope-simulation%20only-lightgrey)
![BSD-3](https://img.shields.io/badge/license-BSD--3--Clause-blue)

<img src="pilots/m2c_handover_recC_20261110/media/demo_fall_getup_walk.gif" width="560" alt="A simulated humanoid is pushed over, gets up on its own, and walks away">

<sub>One illustrative run (single robot, seed 5, 3.5 m/s push, no resets): a push knocks the robot over, the current get-up policy (recC) stands it up, and after the stand check passes the walking policy takes over. Not a measurement; the batch result is 52% of 750 induced falls.</sub>

[Results](docs/RESULTS.md) · [How it works](#how-it-works) · [Quickstart](#quickstart) · [Experiment write-ups](pilots/README.md) · [Glossary](docs/README.md#glossary)

</div>

## At a glance

| Task | Result |
|---|---|
| Walk, stop and turn under random pushes (24 s test) | **237 of 256** robots stay up (93%) |
| Get up from the back, front or side | **68%** of 256 unseen poses pass every check (was 50%) |
| Pushed over, get up, walk away | **52%** of 750 induced falls (was 2.5%) |
| Withstand a shove | Half the robots fall only at a **2.2 m/s** shove, about 3x the largest training push |
| Untrained terrain, blind | At least 80% survival on 19° downhill, 9° uphill and about 6 cm step-downs; fails on step-ups of 4.3 cm and higher |

"Passes every check" means a 2 s stand on the feet with no launch or flight, joint and torque limits respected, a normal stance, and checks frozen before scoring ([EVAL_PROTOCOL.md](EVAL_PROTOCOL.md), [docs/RESULTS.md](docs/RESULTS.md)). **These are single-training-seed results in simulation only.**

## Walking and getting up

<table>
<tr>
<td width="50%" align="center"><img src="walking/media/newton_best_single_robot.gif" alt="The final walking policy, one robot"></td>
<td width="50%" align="center"><img src="recovery/media/recovery_recC_getup_16poses.gif" alt="The current get-up policy standing up 16 robots from lying poses, at half speed"></td>
</tr>
<tr>
<td valign="top"><b>Walking.</b> 237 of 256 robots stayed up through a 24 s stand, walk, turn and stop test with random pushes (231 and 233 in reruns). Most falls happen while turning.</td>
<td valign="top"><b>Getting up</b> <i>(work in progress)</i>. 16 of the unseen lying poses, half speed, no help. The current policy passes every check on 68% of 256 unseen poses.</td>
</tr>
</table>

![Untrained policy (left) and final walking policy (right) on the same scripted commands](walking/media/hero_before_after.gif)

<sub>Walking, before and after training: 16 robots per side, same scripted stand, walk, turn and stop commands. Left: untrained. Right: final walking policy. Fallen robots are reset in place.</sub>

![An earlier policy settling into a forward fold (left) and the previous best policy standing (right), same 16 starting poses](recovery/media/recovery_before_after.gif)

<sub>Where the get-up started: the same 16 lying poses, same camera, no assist. Left: an earlier reward settled into a forward fold. Right: the previous best policy (R9hg), which passed 50%; the current policy above reaches 68%.</sub>

## Stress tests

<table>
<tr>
<td width="33%" align="center"><img src="pilots/m3_terrain_baseline_20261108/media/slope_down_19deg.gif" alt="Eight simulated humanoids walking down a 19 degree slope"></td>
<td width="33%" align="center"><img src="pilots/m3_terrain_baseline_20261108/media/steps_up_4cm.gif" alt="Eight simulated humanoids walking onto 4.3 cm step-ups; most fall"></td>
<td width="33%" align="center"><img src="pilots/m4_robustness_20261108/media/push_2ms.gif" alt="Eight simulated humanoids pushed at 2 metres per second; about half fall"></td>
</tr>
<tr>
<td valign="top"><b>19° downslope.</b> The walking policy, blind, on terrain it never trained on: 8 of 8 robots walked in this run (seed 5; 20 of 24 across the full evaluation, 83%).</td>
<td valign="top"><b>A limitation: 4.3 cm step-ups.</b> 7 of 8 fell in this run (seed 5); across the evaluation 0 of 24 survived at this height. Fallen robots are reset by the environment.</td>
<td valign="top"><b>A 2 m/s shove</b> at 3 s, 8 directions (seed 1): 5 of 8 fell here. Half fall at about 2.2 m/s overall (M50); weakest to the left and rear-left.</td>
</tr>
</table>

<sub>Illustrative single runs of the walking policy (`walking_newton_it1499`), 8 robots each, recorded for this page; the evaluation numbers are in [docs/RESULTS.md](docs/RESULTS.md).</sub>

## What I built

The walking task and PPO are standard Isaac Lab (`Isaac-Velocity-Flat-G1`); the components below are this project's own work. Detail, file by file: [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).

| Component | What it does | Path |
|---|---|---|
| Get-up task package | Environment, rewards, observations, pose curriculum and metrics for training a get-up policy | [`recovery/g1_recovery/`](recovery/g1_recovery/) |
| Validity criteria and protocol | Criteria v3 plus the self-penetration check (3b), frozen before evaluation; DEV / HELDOUT / TEST pose sets | [EVAL_PROTOCOL.md](EVAL_PROTOCOL.md) |
| Hand-over controller | Settle, upper-body blend, readiness gate, command ramp: get-up → walking | [`recovery/g1_recovery/fall_handover.py`](recovery/g1_recovery/fall_handover.py) |
| Scripted walking test | Stand, walk, turn, stop with random pushes, across checkpoints | [`walking/scripts/eval_g1.py`](walking/scripts/eval_g1.py) |
| Stress tests | Push, friction and terrain evaluations with Wilson and bootstrap intervals | [`recovery/scripts/`](recovery/scripts/), [pilots](pilots/m4_robustness_20261108/README.md) |

## How it works

![Recovery system: physics, observations, recovery and walking policies, hand-over, training loop and validity criteria](recovery/architecture/figures/recovery_system_dark.png)

Two policies share one interface: the same 123-value observation in, the same 37 joint-position targets out, and the same network shape, so either can drive the robot. The walking policy comes from Isaac Lab's G1 velocity task, trained with rsl_rl PPO on Newton / MuJoCo-Warp at 50 Hz; the get-up policy uses the same setup, starting from lying poses. On a fall, a debounced detector fires, the get-up policy acts until a criteria-based stand gate passes, and control switches one way to walking through the hand-over controller (settle, upper-body blend, command ramp). Evaluation runs on fixed pose sets — DEV to iterate, HELDOUT to compare, TEST untouched — with validity checks frozen before any result was scored.

## Key findings

- **The get-up still straddles.** Every standing policy here pushes its feet about 1.4 m apart, around 0.4 s into the rise (peak foot spread 1.38 m before, 1.39 m after), before settling to a normal stance. Penalties for foot spread and joint effort cut success from about 128 to 64 to 83 of 256 without moving the spread, so a fix likely needs reference motion ([notes](pilots/humanlike_20261108/ROADMAP.md)).
- **Continued training can collapse because of joint limits.** Policies drifted toward joint limits and fell apart, whatever the learning rate or entropy setting tried. A penalty on body joint limits (weight -100) with a learning rate of 3e-5 kept training stable and produced the recC checkpoint ([Goal 4 write-up](pilots/g4_humanlike_getup_20261108/README.md)).
- **Getting up works; handing over is the weak link.** The get-up policy ends with arms, fingers and waist about 1.2 to 1.3 rad away from the pose the walking policy expects, against 0.03 to 0.07 rad of normal variation, and feet 0.44 m apart against 0.24 m. Blending the upper body before the switch and waiting for a readiness check recover most of the loss ([M2](pilots/m2_fall_handover_20261108/README.md), [M2b](pilots/m2b_handover_fixes_20261108/README.md)).
- **Friction in this solver is a maximum, not a product.** Lowering only the floor friction does nothing, since contacts use the larger of the two materials. The first friction sweep was invalid for this reason and was discarded; the fixed one lowers both together.

## Quickstart

```
git clone https://github.com/kalebcoleman/g1-locomotion-lab && cd g1-locomotion-lab
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-dev.txt
make test
```

`make test` runs the CPU unit tests (metrics, hand-over state machine, robustness statistics); it needs no GPU or Isaac Lab. Training and evaluation need Isaac Lab 3.0.0: see [docs/SETUP.md](docs/SETUP.md).

## Reproduce

- Walking: [train and evaluate](docs/WALKING.md#reproducing). Get-up: [recovery](recovery/README.md). Each pilot ([M2](pilots/m2_fall_handover_20261108/README.md), [M2b](pilots/m2b_handover_fixes_20261108/README.md), [M3](pilots/m3_terrain_baseline_20261108/README.md), [M4](pilots/m4_robustness_20261108/README.md), [M2c](pilots/m2c_handover_recC_20261110/README.md), [G4](pilots/g4_humanlike_getup_20261108/README.md)) ends with its own Reproduce section.

## Explore

| | |
|---|---|
| **Walking write-up** | [Results](docs/WALKING.md#results-at-a-glance) · [method](docs/WALKING.md#method-and-architecture) · [reproduce](docs/WALKING.md#reproducing) · [limitations](docs/WALKING.md#limitations) |
| **Getting up** | [Video gallery](recovery/README.md) · [system overview](recovery/architecture/SYSTEM_OVERVIEW.md) · [network diagram](recovery/architecture/figures/network_architecture_dark.png) · [checkpoints](recovery/checkpoints/README.md) |
| **Status, data, checks** | [Roadmap](pilots/humanlike_20261108/ROADMAP.md) · [status](STATUS.md) · [experiment log](EXPERIMENT_LOG.md) · [walking results](walking/results/) · [GPU physics check](walking/diagnostics/README.md) |

## Repository layout

| Path | Contents |
|---|---|
| `walking/` | The walking project as first published: configs, scripts, results, media, diagnostics, licences |
| `recovery/` | Get-up project: task package `g1_recovery/`, scripts, tests, criteria, media, architecture diagrams ([README](recovery/README.md)) |
| `pilots/` | Dated, bounded experiments, each with its own README ([index](pilots/README.md)): [humanlike](pilots/humanlike_20261108/ROADMAP.md) (negative reward-only pilot and the roadmap), [m2](pilots/m2_fall_handover_20261108/README.md) / [m2b](pilots/m2b_handover_fixes_20261108/README.md) (hand-over), [m3](pilots/m3_terrain_baseline_20261108/README.md) (terrain), [m4](pilots/m4_robustness_20261108/README.md) (pushes, friction), [g4](pilots/g4_humanlike_getup_20261108/README.md) (get-up training, recC) |
| `docs/` | [Walking write-up](docs/WALKING.md), [contributions](docs/CONTRIBUTIONS.md), [results](docs/RESULTS.md), [setup](docs/SETUP.md) |
| `archive/` | Historical experiment outputs: result JSON, run logs and videos from earlier runs ([README](archive/README.md)) |
| `EVAL_PROTOCOL.md`, `EXPERIMENT_LOG.md`, `STATUS.md`, `PROVENANCE.md` | Evaluation protocol (criteria v3), dated experiment log, current status, provenance of imported parts |

## Limitations and next steps

- **The get-up still straddles** (feet ≈1.4 m apart mid-rise); reward-side fixes did not remove it, and a human-like get-up likely needs reference motion.
- **Upward steps fail from 4.3 cm**: all 24 robots fell there, though the policy was trained on flat ground and sees no height scan.
- **One training seed** everywhere; evaluation is not bit-repeatable (walking-test reruns gave 231–237 of 256).
- **Simulation only**: no physical robot, no sim-to-real work.

Next: more training seeds, a less wide get-up, terrain-aware walking.

## Credits and licenses

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by the Isaac Lab Project Developers (NVIDIA and
contributors), the Unitree G1 robot by Unitree Robotics, [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
(ETH Zurich / NVIDIA), [Newton](https://github.com/newton-physics/newton),
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) and NVIDIA Warp. PPO: Schulman et al., 2017.
Licenses: [LICENSE](LICENSE) covers only my original material (BSD-3-Clause, Copyright (c) 2026 Kaleb Coleman); see [NOTICE.md](NOTICE.md).
Third-party code, models and tools keep their own licenses: see [NOTICE.md](walking/NOTICE.md) and the full texts in
[`THIRD_PARTY_LICENSES/`](walking/THIRD_PARTY_LICENSES/). Not affiliated with or endorsed by NVIDIA or Unitree.