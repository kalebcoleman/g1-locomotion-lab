#!/bin/bash
# M2 evaluation batch (evaluation only, no training). Seeds 2-4 (seed 1 was used for calibration), 128 envs, both gates. Each job is logged in ledger.csv (wall seconds).
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; set -u
# recovery policy: the tracked copy of the checkpoint the recorded results used. Override RECOVERY_CKPT to rerun with another one.
RECOVERY_CKPT=${RECOVERY_CKPT:-$ROOT/recovery/checkpoints/R9hg_it1100_previous_baseline.pt}
cd $ROOT/recovery
for GATE in v3 strict; do for SEED in 2 3 4; do
  L=m2_${GATE}_s${SEED}; s=$(date +%s)
  ./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $RECOVERY_CKPT --walk_checkpoint $CK/walking_newton_it1499.pt --num_envs 128 --seed $SEED --levels 0,2.5,3.5,4.5 --gate $GATE --out $P/results/$L.json physics=newton_mjwarp > $P/logs/$L.log 2>&1
  rc=$?; echo "$L,$s,$(date +%s),$(( $(date +%s)-s )),rc=$rc" >> $P/ledger.csv
done; done
echo BATCH_DONE >> $P/ledger.csv
