#!/bin/bash
# Record 5 camera angles (one robot, 8 s: stand/walk/turn/stop, no pushes) into media-raw/<label>/ (gitignored).
# usage: scripts/record_angles.sh <label> <checkpoint> <physics> [extra record_g1.py args]
# env: ISAACLAB_SRC, ISAACLAB_PY as in eval_ladder.sh
set -e
L=$1; CK=$(realpath "$2"); PH=$3; shift 3
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${ISAACLAB_SRC:-$HERE/../IsaacLab}; PY=${ISAACLAB_PY:-python}
export OMNI_KIT_ACCEPT_EULA=YES
cd "$SRC"
$PY "$HERE/scripts/record_g1.py" --task Isaac-Velocity-Flat-G1 --checkpoint "$CK" --label $L \
  --out_dir "$HERE/media-raw/$L" --num_envs 1 --train_env_cfg "$@" physics=$PH
