#!/bin/bash
# M3 terrain baseline (evaluation only): frozen walking it1499 on graded terrain, seeds 1-3, 8 robots per cell. Ledger = wall seconds per job.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; cd $ROOT/recovery
for SEED in ${SEEDS:-1 2 3}; do s=$(date +%s)
  ./scripts/run.sh eval_terrain.py --task Isaac-Velocity-Rough-G1 --walk_checkpoint $CK/walking_newton_it1499.pt --seed $SEED --reps 8 --out $P/results/m3_s$SEED.json physics=newton_mjwarp > $P/logs/m3_s$SEED.log 2>&1
  echo "m3_s$SEED,$s,$(date +%s),$(( $(date +%s)-s )),rc=$?" >> $P/ledger.csv
done; echo DONE >> $P/ledger.csv
