#!/bin/bash
# TEMPLATE, NOT RUN. Fail-loud replacement for the pilot's driver_train.sh. Every gate either runs or aborts the whole sequence; nothing is skipped silently.
set -euo pipefail
P=$(cd "$(dirname "$0")/.." && pwd); PY=$HOME/robotics-workspace/envs/lab-3.0.0/bin/python
LOGS=$HOME/robotics-workspace/envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r; [ -d "$LOGS" ] || { echo "ABORT: logs root missing: $LOGS" >&2; exit 4; }
START=$1; START_SHA=$2; RUN=$3; ITERS=$4                       # explicit arguments; no empty-variable defaults
$PY -I $P/m0/ckpt_inventory.py "$START" --sha256 "$START_SHA" --expect-obs 123 --expect-act 37 > $P/m0/start_inventory.json   # aborts on missing/mismatch
for f in $P/../../recovery/sets/dev_v1.npz $P/../../recovery/sets/heldout_v2.npz; do [ -f "$f" ] || { echo "ABORT: pose set missing $f" >&2; exit 2; }; done
# ... baseline evals via ev.sh (each checks rc) ...
$P/gpu.sh train_main $P/train_pilot.sh "$RUN" 16384 "$ITERS" 21     # gpu.sh must propagate rc (patch: exit $rc)
RD=$($P/m0/find_run.sh "$LOGS" "$RUN")                            # exit 4 if none or several
# gate at it+50: copy and score; failure of any step aborts, then training is stopped by the caller
cp "$RD/model_1150.pt" $P/ckpt/cand_it1150.pt
$PY -I $P/m0/ckpt_inventory.py $P/ckpt/cand_it1150.pt --expect-obs 123 --expect-act 37 > /dev/null
