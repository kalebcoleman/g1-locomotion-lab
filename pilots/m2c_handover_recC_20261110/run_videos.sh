#!/bin/bash
# M2c videos: the same 16 robots as the M2b videos (seed 3, pushes 0/2.5/3.5/4.5 m/s), uncut, recC checkpoint, baseline vs the best v3-gated variant.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; RCK=$ROOT/recovery/checkpoints
REC=$RCK/getup_baseline_recC_it1425.pt; WALK=$CK/walking_newton_it1499.pt; mkdir -p $P/media $P/logs; cd $ROOT/recovery
run(){ L=$1; shift; ./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $REC --walk_checkpoint $WALK --num_envs 16 --seed 3 --levels 0,2.5,3.5,4.5 --record_dir $P/media --label $L --out $P/results/video_$L.json "$@" physics=newton_mjwarp > $P/logs/video_$L.log 2>&1; }
run m2c_recC_base_seed3_16envs
run m2c_recC_blendupper50_seed3_16envs --settle 50 --blend 50 --blend_joints upper
echo VIDEOS_DONE >> $P/ledger.csv
