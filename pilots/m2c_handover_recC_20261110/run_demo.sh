#!/bin/bash
# M2c portfolio demo recording: 8 robots (one per push magnitude in the rotating list), fixed close camera on the robot that starts at the env-0 origin.
# The protocol and variant flags are the M2b/M2c ones; --walk_s/--total_s are the standard values here, the clip is trimmed later by make_demo_gif.sh.
# usage: run_demo.sh [label]    (default label demo_fall_getup_walk; this is the run behind media/demo_fall_getup_walk.gif)
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; RCK=$ROOT/recovery/checkpoints
REC=$RCK/getup_baseline_recC_it1425.pt; WALK=$CK/walking_newton_it1499.pt
LABEL=${1:-demo_fall_getup_walk}
mkdir -p $P/media $P/logs; cd $ROOT/recovery
./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $REC --walk_checkpoint $WALK \
  --num_envs 8 --seed 3 --levels 3.5,3.5,4.5,4.5,2.5,2.5,3.5,4.5 --walk_s 6 --total_s 23 \
  --settle 50 --blend 50 --blend_joints upper \
  --record_dir $P/media --label $LABEL --out $P/results/$LABEL.json \
  --eye 2.5,-3.0,1.4 --lookat 0,0,0.25 --cam_focal 30 \
  physics=newton_mjwarp > $P/logs/$LABEL.log 2>&1
echo "$LABEL,seed=3,envs=8,rc=$?" >> $P/ledger.csv
