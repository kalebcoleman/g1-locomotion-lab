#!/bin/bash
# usage: evalv2.sh <label> <task> <checkpoint> <poseset.npz> [assist_level]   -> results/<label>.json (v2 protocol), repeatable via different labels
ROOT=$(cd "$(dirname "$0")/../.." && pwd); L=$1; T=$2; CK=$3; SET=$4; AS=$5
[ -n "$AS" ] && export G1R_EVAL_ASSIST=$AS
cd $ROOT/recovery && ./scripts/gpu_run.py --name ev_$L --est_gb ${G1R_EST_GB:-4} --settle_s ${G1R_SETTLE_S:-15} --log $ROOT/archive/runs/ev_$L.log -- bash -c "cd $ROOT/recovery && ./scripts/run.sh eval_recovery.py --task $T --checkpoint $CK --num_envs 256 --heldout $SET --out $ROOT/archive/results/$L.json physics=newton_mjwarp" | tail -1
