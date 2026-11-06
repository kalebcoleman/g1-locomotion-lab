#!/bin/bash
# Record the 16-robot progression clips (one 10 s MP4 per checkpoint) into media-raw/progression/.
# usage: scripts/record_progression.sh <run_dir>   (checkpoints 0 50 100 200 350 500 700 1000 1499)
set -e
RUN=$(realpath "$1")
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=${ISAACLAB_SRC:-$HERE/../IsaacLab}; PY=${ISAACLAB_PY:-python}
export OMNI_KIT_ACCEPT_EULA=YES
cd "$SRC"
$PY "$HERE/scripts/record_prog.py" --task Isaac-Velocity-Flat-G1 --run_dir "$RUN" \
  --ckpts 0,50,100,200,350,500,700,1000,1499 --label prog --out_dir "$HERE/media-raw/progression" \
  --num_envs 16 --train_env_cfg physics=newton_mjwarp
