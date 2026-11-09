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
<td valign="top"><b>Getting up</b> <i>(work in progress)</i>. From lying on the back, front or side, with no help, in about a second. The best policy passes strict validity checks on about half of held-out starting poses.</td>
</tr>
</table>

## From a fold to a stand

![Earlier policy stuck in a forward fold (left) and the current best policy standing (right), same 16 starting poses](recovery/media/recovery_before_after.gif)

<sub>Same 16 lying starts, same camera, no assist. Left: an earlier reward settled into a forward fold, pelvis high and torso upside down. Right: the current best policy stands up.</sub>

## At a glance

| | Result | Notes |
|---|---|---|
| **Walking** | 237 / 256 robots stay up | 24 s scripted test with pushes; reruns 231 and 233; one seed |
| **Getting up** | 49.6-50.4% of 256 held-out poses pass all checks | three evaluations, assist off; prone about 78%, side B about 27%; one seed |
| **What "passes" means** | supported 2 s stand, no launch or flight, joint, torque and self-penetration limits, normal stance, still up at the end | details in [recovery/README.md](recovery/README.md) |
| **Not done yet** | hand-over from getting up to walking; replication on more seeds | training now |

## Architecture

![Recovery system: physics, observations, recovery and walking policies, hand-over, training loop and validity criteria](recovery/architecture/figures/recovery_system_dark.png)

## Explore

| | |
|---|---|
| **Walking write-up** | [Results](docs/WALKING.md#results-at-a-glance) · [method](docs/WALKING.md#method-and-architecture) · [reproduce](docs/WALKING.md#reproducing) · [limitations](docs/WALKING.md#limitations) |
| **Getting up** | [Video gallery](recovery/README.md) · [system overview](recovery/architecture/SYSTEM_OVERVIEW.md) · [network diagram](recovery/architecture/figures/network_architecture_dark.png) |
| **GPU physics check** | [Newton / MuJoCo-Warp kernels run on the GPU](walking/diagnostics/README.md) |
| **Data** | [Results](walking/results/) · [configs](walking/configs/) · [scripts](walking/scripts/) |

## Repository layout

| Path | Contents |
|---|---|
| `walking/` | The walking project as first published: `configs/`, `scripts/`, `results/`, `media/`, `diagnostics/`, licences |
| `recovery/` | Get-up experiments: task package `g1_recovery/`, `scripts/`, `tests/`, criteria, report, research notes, small checkpoints, media and architecture diagrams ([README](recovery/README.md)) |
| `pilots/humanlike_20261108/` | One reward-regularised warm-start pilot (negative result), reproducibility tooling and the staged roadmap ([ROADMAP](pilots/humanlike_20261108/ROADMAP.md), [HANDOFF](pilots/humanlike_20261108/HANDOFF.md)) |
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
