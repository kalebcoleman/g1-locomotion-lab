<div align="center">

# G1 Locomotion Lab

**A Unitree G1 humanoid learning to walk, and to get back up after a fall, in simulation.**

![Isaac Lab 3.0.0](https://img.shields.io/badge/Isaac%20Lab-3.0.0-76B900)
![Physics](https://img.shields.io/badge/physics-Newton%20%2F%20MuJoCo--Warp-1f6feb)
![Robot](https://img.shields.io/badge/robot-Unitree%20G1-555)
![RL](https://img.shields.io/badge/RL-PPO%20(rsl__rl)-8250df)
![Scope](https://img.shields.io/badge/scope-simulation%20only-lightgrey)
![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)

![Unitree G1 policies at training iteration 0 (left) and 1499 (right), same scripted commands](walking/media/hero_before_after.gif)

<sub>16 robots per side, same scripted stand, walk, turn and stop commands. Left: untrained. Right: final walking policy. Fallen robots are reset in place.</sub>

</div>

## Walking and getting up

<table>
<tr>
<td width="50%" align="center"><img src="walking/media/newton_best_single_robot.gif" alt="Final walking policy, one robot"></td>
<td width="50%" align="center"><img src="recovery/media/recovery_best_close.gif" alt="Best get-up policy, four lying starts"></td>
</tr>
<tr>
<td valign="top"><b>Walking.</b> 237 of 256 robots stayed up through a 24 s stand, walk, turn and stop test with random pushes (231 and 233 in reruns). Most falls happen while turning.</td>
<td valign="top"><b>Getting up</b> <i>(work in progress)</i>. From lying on the back, front or side, with no help, in about a second. The current baseline passes strict validity checks on 68% of held-out starting poses.</td>
</tr>
</table>

## From a fold to a stand

![Earlier policy stuck in a forward fold (left) and the current best policy standing (right), same 16 starting poses](recovery/media/recovery_before_after.gif)

<sub>Same 16 lying starts, same camera, no assist. Left: an earlier reward settled into a forward fold, pelvis high and torso upside down. Right: a get-up policy standing up.</sub>

## At a glance

| | Result | Notes |
|---|---|---|
| **Walking** | 237 / 256 robots stay up | 24 s scripted test with pushes; reruns 231 and 233; one seed |
| **Getting up** | **68.2%** of 256 held-out poses pass all checks (mean 174.7 of 256), up from 50.5% (129.3) | checkpoint `recC`, assist off, one seed; by direction (of 64 each) side A 52, prone 52, side B 36, supine 35, against 31, 47, 19, 33 before |
| **Hand-over, getting up to walking** | 6 of 250 robots pass the strict check when handed to the walking policy straight away; **28.7%** with a blend of the upper body; **99.9%** when the hand-over waits for a readiness check instead | zero re-falls with the blend; the readiness gate is not the same test as the strict check ([M2b](pilots/m2b_handover_fixes_20261108/README.md)) |
| **Pushes** | half of the robots fall at about 2.2 m/s of added speed (M50 2.16 m/s) | training used pushes up to 0.5 m/s; weakest to the left and rear-left (about 1.4 m/s), strongest forward-right (3.4 to 3.6) ([M4](pilots/m4_robustness_20261108/README.md)) |
| **Low friction** | M50 falls from 2.16 m/s at friction 1.0 to 1.45 at 0.3 | walking alone survives 98-99% of the first 3 s down to 0.4 and 56% at 0.2 |
| **Terrain** | at least 80% survival on 19 degree downhill, 9 degree uphill, 5.8 cm steps down, 2.8 cm steps up, 8 cm rough noise | blind walking, no height scan; the policy was trained on flat ground ([M3](pilots/m3_terrain_baseline_20261108/README.md)) |
| **What "passes" means** | supported 2 s stand, no launch or flight, joint, torque and self-penetration limits, normal stance, still up at the end | details in [recovery/README.md](recovery/README.md) |

All get-up numbers use criteria v3 before the self-penetration check, on the held-out set. The test set is untouched.

## What we learned

- **The get-up still straddles.** Every standing policy here pushes its feet about 1.4 m apart, around 0.4 s into the rise (peak foot spread 1.38 m before, 1.39 m after), before settling to a normal stance. Reward-only changes did not remove it. A pilot that added penalties for foot spread and joint effort lowered success from about 128 to 64 to 83 of 256 and left the spread unchanged. Looking human-like is not measured by any check, and a stronger fix likely needs reference motion ([notes](pilots/humanlike_20261108/ROADMAP.md)).
- **Continued training can collapse because of joint limits.** Policies drifted toward joint limits and fell apart. A penalty on body joint limits (weight -100) with a learning rate of 3e-5 kept training stable and produced the recC checkpoint ([Goal 4 write-up](pilots/g4_humanlike_getup_20261108/README.md)).
- **Getting up works; handing over is the weak link.** The get-up policy ends with arms, fingers and waist about 1.2 to 1.3 rad away from the pose the walking policy was trained on, against 0.03 to 0.07 rad of variation in normal walking, and feet 0.44 m apart against 0.24 m. Handing over immediately fails; blending the upper body and waiting for a readiness check recover most of the loss ([M2](pilots/m2_fall_handover_20261108/README.md), [M2b](pilots/m2b_handover_fixes_20261108/README.md)).
- **Friction in this solver is a maximum, not a product.** Lowering only the floor friction does nothing, since contacts use the larger of the two materials. The first friction sweep was invalid for this reason and was discarded; the fixed one lowers both together.

## Architecture

![Recovery system: physics, observations, recovery and walking policies, hand-over, training loop and validity criteria](recovery/architecture/figures/recovery_system_dark.png)

## Explore

| | |
|---|---|
| **Walking write-up** | [Results](docs/WALKING.md#results-at-a-glance) · [method](docs/WALKING.md#method-and-architecture) · [reproduce](docs/WALKING.md#reproducing) · [limitations](docs/WALKING.md#limitations) |
| **Getting up** | [Video gallery](recovery/README.md) · [system overview](recovery/architecture/SYSTEM_OVERVIEW.md) · [network diagram](recovery/architecture/figures/network_architecture_dark.png) · [get-up checkpoints](recovery/checkpoints/README.md) |
| **Roadmap and status** | [Roadmap](pilots/humanlike_20261108/ROADMAP.md) · [status](STATUS.md) · [experiment log](EXPERIMENT_LOG.md) |
| **GPU physics check** | [Newton / MuJoCo-Warp kernels run on the GPU](walking/diagnostics/README.md) |
| **Data** | [Results](walking/results/) · [configs](walking/configs/) · [scripts](walking/scripts/) |

## Repository layout

| Path | Contents |
|---|---|
| `walking/` | The walking project as first published: `configs/`, `scripts/`, `results/`, `media/`, `diagnostics/`, licences |
| `recovery/` | Get-up experiments: task package `g1_recovery/`, `scripts/`, `tests/`, criteria, report, research notes, small checkpoints including the get-up baseline, media and architecture diagrams ([README](recovery/README.md)) |
| `pilots/` | Dated, bounded experiments, each with its own README, results and ledger: [humanlike_20261108](pilots/humanlike_20261108/ROADMAP.md) (reward-only pilot, negative, and the roadmap), [m2](pilots/m2_fall_handover_20261108/README.md) and [m2b](pilots/m2b_handover_fixes_20261108/README.md) (fall, get-up, walk hand-over), [m3](pilots/m3_terrain_baseline_20261108/README.md) (terrain), [m4](pilots/m4_robustness_20261108/README.md) (pushes, friction), [g4](pilots/g4_humanlike_getup_20261108/README.md) (get-up training, recC) |
| `EVAL_PROTOCOL.md`, `EXPERIMENT_LOG.md`, `STATUS.md`, `PROVENANCE.md` | Evaluation criteria (v3), dated experiment log, current status, where each imported part came from |
| `results/`, `runs/`, `media/`, `checkpoints/` | Recovery-project result JSON, run logs, videos and the walking checkpoint (the walking project keeps its own copies under `walking/`) |

## Credits and licenses

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by the Isaac Lab Project Developers (NVIDIA and
contributors), the Unitree G1 robot by Unitree Robotics, [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
(ETH Zurich / NVIDIA), [Newton](https://github.com/newton-physics/newton),
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) and NVIDIA Warp. PPO: Schulman et al., 2017.
Licenses: [LICENSE](walking/LICENSE) covers only my original material (BSD-3-Clause, Copyright (c) 2026 Kaleb Coleman).
Third-party code, models and tools keep their own licenses: see [NOTICE.md](walking/NOTICE.md) and the full texts in
[`THIRD_PARTY_LICENSES/`](walking/THIRD_PARTY_LICENSES/). Not affiliated with or endorsed by NVIDIA or Unitree.
