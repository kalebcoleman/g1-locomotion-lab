# G1 Locomotion Lab

**Training and evaluating a Unitree G1 humanoid in simulation: walking, and getting back up after a fall.**
Isaac Lab 3.0.0 with the Newton / MuJoCo-Warp physics backend. Simulation only, no physical robot.

![Unitree G1 policies at training iteration 0 (left) and 1499 (right), same scripted commands](media/hero_before_after.gif)

*16 robots per side, same scripted stand, walk, turn and stop commands. Left: untrained policy. Right: final walking policy. Fallen robots are reset in place.*

## Highlights

<table>
<tr>
<td width="50%"><img src="media/newton_best_single_robot.gif" alt="Final walking policy, one robot"><br><b>Walking.</b> 237 of 256 robots stayed up through a 24 s stand, walk, turn and stop test with random pushes (231 and 233 in reruns). Turning causes most falls.</td>
<td width="50%"><img src="recovery/media/20261107_03_R9hg_it1100_close_sheet.png" alt="Best recovery policy so far"><br><b>Getting up (work in progress).</b> The best policy so far stands up unassisted from 49.6-50.4% of 256 held-out lying poses under strict validity checks. One seed; uneven across fall directions.</td>
</tr>
<tr>
<td><img src="media/progression_chart.png" alt="Walking training progression"><br>Walking: training log and scripted test across checkpoints.</td>
<td><img src="recovery/architecture/figures/recovery_system_dark.png" alt="Recovery system architecture"><br>Recovery system: policies, hand-over, training loop and validity criteria.</td>
</tr>
</table>

## Explore

| | |
|---|---|
| **Walking write-up** | [Results](docs/WALKING.md#results-at-a-glance), [method and architecture](docs/WALKING.md#method-and-architecture), [how to reproduce](docs/WALKING.md#reproducing), [limitations](docs/WALKING.md#limitations) |
| **Fall recovery** | [Videos, pictures and architecture](recovery/README.md), [system overview](recovery/architecture/SYSTEM_OVERVIEW.md) |
| **GPU-physics check** | [Newton / MuJoCo-Warp kernels run on the GPU](diagnostics/README.md) |
| **Data** | [Evaluation results](results/), [resolved configs](configs/), [scripts](scripts/) |

## Credits and licenses

Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) by the Isaac Lab Project Developers (NVIDIA and
contributors), the Unitree G1 robot by Unitree Robotics, [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
(ETH Zurich / NVIDIA), [Newton](https://github.com/newton-physics/newton),
[MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) and NVIDIA Warp. PPO: Schulman et al., 2017.
Licenses: [LICENSE](LICENSE) covers only my original material (BSD-3-Clause, Copyright (c) 2026 Kaleb Coleman).
Third-party code, models and tools keep their own licenses: see [NOTICE.md](NOTICE.md) and the full texts in
[`THIRD_PARTY_LICENSES/`](THIRD_PARTY_LICENSES/). Not affiliated with or endorsed by NVIDIA or Unitree.
