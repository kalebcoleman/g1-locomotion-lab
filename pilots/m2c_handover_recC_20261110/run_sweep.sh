#!/bin/bash
# M2c full sweep: 3 variants x (3 independent repeats on seeds 2-4 + one pass on fresh seeds 5-7) = 36 jobs, run sequentially.
# usage: run_sweep.sh            (the r1/base/s2 job is skipped if its result already exists)
P=$(cd "$(dirname "$0")" && pwd); cd $P
[ -f results/m2c_base_r1_s2.json ] || ./run.sh r1 2 base
./run.sh r1 3,4 base; ./run.sh r2 2,3,4 base; ./run.sh r3 2,3,4 base; ./run.sh fresh 5,6,7 base
for V in blendupper50 ready_upper50; do
  ./run.sh r1 2,3,4 $V; ./run.sh r2 2,3,4 $V; ./run.sh r3 2,3,4 $V; ./run.sh fresh 5,6,7 $V
done
echo SWEEP_DONE >> $P/ledger.csv
