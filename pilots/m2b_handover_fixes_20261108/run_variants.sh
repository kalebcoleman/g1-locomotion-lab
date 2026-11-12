#!/bin/bash
# M2b: eval-only hand-over variants (no training). usage: run_variants.sh <seeds> <repeat-tag> <variant>...   variants are looked up in the table below.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints
# recovery policy: the tracked copy of the checkpoint the recorded results used. Override RECOVERY_CKPT to rerun with another one.
RECOVERY_CKPT=${RECOVERY_CKPT:-$ROOT/recovery/checkpoints/R9hg_it1100_previous_baseline.pt}; cd $ROOT/recovery
declare -A V=(
 [base]=""
 [settle25]="--settle 25" [settle50]="--settle 50" [settle100]="--settle 100"
 [hold50]="--settle 50 --settle_mode hold"
 [blendall50]="--settle 50 --blend 50 --blend_joints all"
 [blendhands50]="--settle 50 --blend 50 --blend_joints hands"
 [blendupper50]="--settle 50 --blend 50 --blend_joints upper"
 [blendupper25]="--settle 25 --blend 25 --blend_joints upper"
 [ramp50]="--ramp 50" [ramp100]="--ramp 100"
 [ready]="--trigger ready" [ready_settle50]="--trigger ready --settle 50" [ready_upper50]="--trigger ready --settle 50 --blend 50 --blend_joints upper"
 [upper50_ramp50]="--settle 50 --blend 50 --blend_joints upper --ramp 50"
 [ready_upper50_ramp50]="--trigger ready --settle 50 --blend 50 --blend_joints upper --ramp 50"
)
SEEDS=$1; TAG=$2; shift 2
for VAR in "$@"; do for SEED in $SEEDS; do
  L=m2b_${VAR}${TAG}_s${SEED}; s=$(date +%s)
  ./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $RECOVERY_CKPT --walk_checkpoint $CK/walking_newton_it1499.pt --num_envs 128 --seed $SEED --levels 0,2.5,3.5,4.5 ${V[$VAR]} --out $P/results/$L.json physics=newton_mjwarp > $P/logs/$L.log 2>&1
  echo "$L,$s,$(date +%s),$(( $(date +%s)-s )),rc=$?,${V[$VAR]}" >> $P/ledger.csv
done; done; echo "DONE $TAG" >> $P/ledger.csv
