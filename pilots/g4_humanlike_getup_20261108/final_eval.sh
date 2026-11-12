#!/bin/bash
# Goal 4 final evaluation (evaluation only): baseline and a candidate checkpoint, HELDOUT x3 and DEV x1 with the movement metrics, then matched videos.
# usage: final_eval.sh <candidate.pt> <tag>
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); R=$ROOT/recovery; CAND=$1; TAG=$2
BASE=${RECOVERY_CKPT:-$ROOT/recovery/checkpoints/R9hg_it1100_previous_baseline.pt}   # override to compare against another baseline
cd $R
ev(){ L=$1; CK=$2; SET=$3; ./scripts/run.sh eval_recovery.py --task G1-Recovery-R9hg-v0 --checkpoint $CK --num_envs 256 --heldout $R/sets/$SET --out $P/results/final_${L}.json physics=newton_mjwarp > $P/logs/final_${L}.log 2>&1; echo "final_$L,$(date +%s),rc=$?" >> $P/ledger_eval.csv; }
for i in 1 2 3; do ev base_heldout_r$i $BASE heldout_v2.npz; ev ${TAG}_heldout_r$i $CAND heldout_v2.npz; done
ev base_dev $BASE dev_v1.npz; ev ${TAG}_dev $CAND dev_v1.npz
rec(){ L=$1; CK=$2; ./scripts/run.sh record_recovery.py --task G1-Recovery-R9hg-v0 --checkpoint $CK --out_dir $P/media --label $L --num_envs 16 --heldout $R/sets/heldout_v2.npz physics=newton_mjwarp > $P/logs/video_$L.log 2>&1; }
[ -z "$NOVIDEO" ] && { rec g4_base_heldout16 $BASE; rec g4_${TAG}_heldout16 $CAND; }
echo FINAL_DONE_$TAG >> $P/ledger_eval.csv
