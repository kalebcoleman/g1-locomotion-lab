#!/bin/bash
# One short warm-started training stage of the R9hg recovery task. usage: train_stage.sh <run_name> <num_envs> <iters> <seed> <init_ckpt> [hydra overrides...]
# The init checkpoint is only READ (originals stay preserved); the optimizer is reset (weights + iteration counter restored), see EXPERIMENT_LOG / m0 run manifest.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); LAB=$HOME/robotics-workspace/envs/lab-3.0.0-src; PY=$HOME/robotics-workspace/envs/lab-3.0.0/bin/python
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$ROOT/recovery${PYTHONPATH:+:$PYTHONPATH}
NAME=$1; NE=$2; IT=$3; SEED=$4; INIT=$5; shift 5
[ -f "$INIT" ] || { echo "init checkpoint missing: $INIT" >&2; exit 2; }
cd $LAB && exec $PY scripts/reinforcement_learning/train.py --task G1-Recovery-R9hg-v0 --num_envs $NE --rl_library rsl_rl --max_iterations $IT --seed $SEED --run_name $NAME --viz none \
  --external_callback=g1_recovery.register.register --checkpoint $INIT --no-resume_optimizer_state agent.save_interval=${SAVE_INTERVAL:-25} "$@"
