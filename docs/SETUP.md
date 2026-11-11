# Setup

Two independent setups: the CPU tests (everything CI needs) and the simulation (Isaac Lab, GPU only). The GitHub
Actions workflow runs only the first one.

## CPU tests

The unit tests are plain Python scripts. They need Python 3.12, `torch`, `numpy` and `gymnasium`; they never import
Isaac Lab, Warp, MuJoCo or Isaac Sim.

```bash
python3 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements-dev.txt
make test
```

`make test` runs the four test scripts and stops at the first failure. Pass another interpreter with
`make test PYTHON=/path/to/python`. `requirements-dev.txt` pins the versions the repository was tested with
(Python 3.12, numpy 2.5.4, torch 2.11.0, gymnasium 1.4.0).

`make check-links` checks every relative link and image in the tracked Markdown files (`tools/check_md_links.py`,
no dependencies). It does not check the links inside code blocks or anchors within a page.

## Simulation: Isaac Lab 3.0.0 with Newton / MuJoCo-Warp

The simulation is **not pip-installable** and is not needed for the tests. It requires Linux, an NVIDIA GPU with a
recent driver, and a local Isaac Lab 3.0.0 checkout. The runs in this repository used Isaac Lab 3.0.0 from commit
`e3f7d9f` (see [WALKING.md](WALKING.md#reproducing)) on an RTX PRO 6000 Blackwell (98 GB, shared). Install Isaac Lab by
following its [installation guide](https://isaac-sim.github.io/IsaacLab/); the Unitree G1 assets are fetched by Isaac
Lab at run time and are not redistributed here.

Every command passes the Hydra override `physics=newton_mjwarp`, which selects the Newton / MuJoCo-Warp physics
backend instead of PhysX. Kit needs `OMNI_KIT_ACCEPT_EULA=YES`; `recovery/scripts/run.sh` exports it.

### How `recovery/scripts/run.sh` finds Isaac Lab

The wrapper is deliberately small and expects one directory layout:

```bash
ROOT=$(cd "$(dirname "$0")/.." && pwd)       # <repo>/recovery
export PYTHONPATH=$ROOT:$PYTHONPATH          # makes g1_recovery importable
cd $HOME/isaac-workspace/envs/lab-3.0.0-src  # Isaac Lab source checkout
exec ../lab-3.0.0/bin/python "$ROOT/scripts/$1" "${@:2}"
```

So it expects a **source checkout** at `~/isaac-workspace/envs/lab-3.0.0-src` and its **Python environment** at
`~/isaac-workspace/envs/lab-3.0.0` (the two are siblings; the interpreter is `../lab-3.0.0/bin/python`). If your
checkout lives elsewhere, edit those two paths in `recovery/scripts/run.sh`, or create the same layout with a symlink.
All arguments after the script name are passed through.

A get-up evaluation with the current baseline:

```bash
cd recovery
./scripts/run.sh eval_recovery.py --task G1-Recovery-R9hg-v0 \
    --checkpoint checkpoints/getup_baseline_recC_it1425.pt \
    --num_envs 256 --heldout sets/heldout_v2.npz --out /tmp/eval.json physics=newton_mjwarp
```

Training the walking policy uses Isaac Lab's own trainer from the checkout; the exact command is in
[WALKING.md](WALKING.md#reproducing). The pilots wrap the same pattern for their own experiments (for example
`pilots/m2_fall_handover_20261108/run_batch.sh`), and all of them accept the environment overrides documented in
their README.

Checkpoints are large binary files and are gitignored by default. The small, referenced baselines are tracked:
`checkpoints/walking_newton_it1499.pt` (walking) and `recovery/checkpoints/*.pt` (get-up).
