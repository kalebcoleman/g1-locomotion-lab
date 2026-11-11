#!/bin/bash
# M2c: M2b hand-over evaluation rerun with the recC get-up baseline (getup_baseline_recC_it1425.pt). No training, no weight changes.
# The protocol, task, pushes, walking policy and variant flags are identical to M2b; only --checkpoint differs.
# usage: run.sh <run-tag> <seeds> <variant>...     (variant names are looked up in the table below; run-tag r1/r2/r3/fresh)
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; RCK=$ROOT/recovery/checkpoints
REC=$RCK/getup_baseline_recC_it1425.pt; WALK=$CK/walking_newton_it1499.pt
mkdir -p $P/results $P/logs
cd $ROOT/recovery
declare -A V=(
 [base]=""
 [blendupper50]="--settle 50 --blend 50 --blend_joints upper"
 [ready_upper50]="--trigger ready --settle 50 --blend 50 --blend_joints upper"
)
RUN=$1; SEEDS=${2//,/ }; shift 2
for VAR in "$@"; do for SEED in $SEEDS; do
  L=m2c_${VAR}_${RUN}_s${SEED}; s=$(date +%s)
  ./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $REC --walk_checkpoint $WALK --num_envs 128 --seed $SEED --levels 0,2.5,3.5,4.5 ${V[$VAR]} --out $P/results/$L.json physics=newton_mjwarp > $P/logs/$L.log 2>&1
  rc=$?
  echo "$L,$s,$(date +%s),$(( $(date +%s)-s )),rc=$rc,${V[$VAR]}" >> $P/ledger.csv
done; done; echo "DONE $RUN $*" >> $P/ledger.csv
