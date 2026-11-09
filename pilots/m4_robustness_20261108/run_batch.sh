#!/bin/bash
# M4 push/friction robustness (evaluation only): frozen walking it1499, seeds 1-3, 4 robots per (magnitude, direction) cell, floor friction sweep. Ledger = wall seconds per job.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; cd $ROOT/recovery
for MU in ${MUS:-1.0 0.8 0.6 0.4 0.3 0.2 0.1}; do for SEED in ${SEEDS:-1 2 3}; do s=$(date +%s)
  ./scripts/run.sh eval_robustness.py --task Isaac-Velocity-Flat-G1 --walk_checkpoint $CK/walking_newton_it1499.pt --seed $SEED --reps 4 --friction $MU --out $P/results/m4_mu${MU}_s$SEED.json physics=newton_mjwarp > $P/logs/m4_mu${MU}_s$SEED.log 2>&1
  echo "m4_mu${MU}_s$SEED,$s,$(date +%s),$(( $(date +%s)-s )),rc=$?" >> $P/ledger.csv
done; done; echo DONE >> $P/ledger.csv
