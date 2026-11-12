# Contributions

What came from upstream and what this project built. Back to the [overview](../README.md).

## Built on

| Piece | Origin |
|---|---|
| Isaac Lab 3.0.0 | The `Isaac-Velocity-Flat-G1` task: environment, rewards, observations, terminations and the rsl_rl PPO hyperparameters for walking. Unmodified. |
| rsl_rl PPO | The algorithm (Schulman et al., 2017) and its `OnPolicyRunner` implementation (ETH Zurich / NVIDIA). |
| Newton / MuJoCo-Warp | Physics backend (`physics=newton_mjwarp`), on top of NVIDIA Warp and Isaac Sim. |
| Unitree G1 robot model | Fetched by Isaac Lab at run time; kept its own licence (see [NOTICE.md](../walking/NOTICE.md)). |

For walking I only chose the backend, environment count (1,024), seed (1) and the default iteration count (1,500), then wrote the scripted evaluation. The task, rewards and hyperparameters are untouched ([details](WALKING.md#summary)).

## Built here

First person, file by file.

- **The get-up task package** — [`recovery/g1_recovery/`](../recovery/g1_recovery/): I wrote the environment config (`env_cfg.py`: the full-collision G1 base; the Unitree torque limits and self-collision come from the R9hc / R8wc / R9hg configs in `lane_a.py`), the reward terms (`lane_a.py`, `lane_b.py`), the observation setup, the pose curriculum and run registration (`register.py`), the training assist (`assist.py`), and the success metrics (`metrics.py`: supported stand, plausibility, chain success).
- **The validity criteria and evaluation protocol** — [EVAL_PROTOCOL.md](../EVAL_PROTOCOL.md): I wrote criteria v3 (v2.1 plus Amendment 3, including the torque rule 3a, the self-penetration screen 3b, persistence 3c and the stance rule 3d), froze them before evaluation, and defined the DEV / HELDOUT / TEST pose sets with the rule that TEST stays untouched until a final claim. The earlier v1 criteria are in [recovery/CRITERIA.md](../recovery/CRITERIA.md).
- **The fall → get-up → walk hand-over controller** — [`recovery/g1_recovery/fall_handover.py`](../recovery/g1_recovery/fall_handover.py): a one-way state machine with debounced fall detection, a settle window, an upper-body (arms and fingers) blend to the default stance, a readiness gate as an alternative trigger, and a forward-command ramp after the switch. Driven by [`recovery/scripts/eval_fall_handover.py`](../recovery/scripts/eval_fall_handover.py); 24 CPU unit tests.
- **The scripted walking evaluation** — [`walking/scripts/eval_g1.py`](../walking/scripts/eval_g1.py) plus the ladder and recording scripts: a 24 s stand / walk / turn / stop test with random pushes across checkpoints, in 256 fresh environments.
- **The stress tests and their statistics** — push and friction ([`recovery/scripts/eval_robustness.py`](../recovery/scripts/eval_robustness.py)) and terrain ([`recovery/scripts/eval_terrain.py`](../recovery/scripts/eval_terrain.py)), pooled with Wilson intervals and bootstrap fits ([`recovery/g1_recovery/robustness_stats.py`](../recovery/g1_recovery/robustness_stats.py)), reported in the [M3](../pilots/m3_terrain_baseline_20261108/README.md) and [M4](../pilots/m4_robustness_20261108/README.md) pilots.
- **The experiment programme** — the dated, bounded pilots under [`pilots/`](../pilots/): hand-over baseline and fixes ([M2](../pilots/m2_fall_handover_20261108/README.md), [M2b](../pilots/m2b_handover_fixes_20261108/README.md)), terrain (M3), robustness (M4), the reward-only human-like pilot (negative, with the roadmap) and the get-up training line that produced the recC checkpoint ([G4](../pilots/g4_humanlike_getup_20261108/README.md)).
- **Tests** — [`recovery/tests/test_metrics.py`](../recovery/tests/test_metrics.py) (12 metric tests, required before any result is trusted) and the hand-over state-machine tests.
- **Write-ups and diagrams** — [docs/WALKING.md](WALKING.md), the [system overview](../recovery/architecture/SYSTEM_OVERVIEW.md) and the architecture diagrams in [`recovery/architecture/`](../recovery/architecture/).

Results from all of the above, with caveats: [docs/RESULTS.md](RESULTS.md).