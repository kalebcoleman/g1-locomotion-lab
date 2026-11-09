#!/bin/bash
# Evaluate checkpoints of one training run with the scripted stand/walk/turn/stop test.
# usage: scripts/eval_ladder.sh <run_dir> <physics> <iter...>
#   run_dir  : path to an rsl_rl run folder containing model_<iter>.pt
#   physics  : newton_mjwarp (the backend used for the published results)
# env: ISAACLAB_SRC (Isaac Lab checkout, default ../IsaacLab), ISAACLAB_PY (python of the Isaac Lab env, default python)
set -e
RUN=$(realpath "$1"); PH=$2; shift 2
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${ISAACLAB_SRC:-$HERE/../IsaacLab}; PY=${ISAACLAB_PY:-python}
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$HERE/results/raw"
cd "$SRC"
for it in "$@"; do
  $PY "$HERE/scripts/eval_g1.py" --task Isaac-Velocity-Flat-G1 \
    --checkpoint "$RUN/model_$it.pt" --num_envs 256 --train_env_cfg --push_vel 0.5 --push_interval 2.5 \
    --out "$HERE/results/raw/$(basename "$RUN")_${PH}_model${it}_push0.5.json" physics=$PH
  echo "done $it"
done
