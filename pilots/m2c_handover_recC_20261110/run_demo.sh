#!/bin/bash
# M2c portfolio hero recording: ONE robot, no resets, fixed wide camera (the recorder has no tracking camera).
# usage: run_demo.sh [label]    (default label portfolio_hero1_s5, the recording behind media/demo_fall_getup_walk.gif)
# Protocol: recC get-up policy, blendupper50 hand-over (settle 1 s + upper-body blend), walking it1499; push 3.5 m/s at 1.5 s; seed 5 (chosen after scouting seeds 1-4 and a
# seed-3 8-robot run; see the README). Run from anywhere; the video is written to media/ (gitignored) and the result JSON to results/.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints; RCK=$ROOT/recovery/checkpoints
REC=$RCK/getup_baseline_recC_it1425.pt; WALK=$CK/walking_newton_it1499.pt
LABEL=${1:-portfolio_hero1_s5}
mkdir -p $P/media $P/logs $P/results; cd $ROOT/recovery
./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $REC --walk_checkpoint $WALK \
  --num_envs 1 --seed 5 --levels 3.5 --push_s 1.5 --walk_s 5 --total_s 15 \
  --settle 50 --blend 50 --blend_joints upper \
  --record_dir $P/media --label $LABEL --out $P/results/$LABEL.json \
  --eye 0,-8,4.5 --lookat 0.5,0,0.5 --cam_focal 12 \
  physics=newton_mjwarp > $P/logs/$LABEL.log 2>&1
echo "$LABEL,seed=5,envs=1,rc=$?" >> $P/ledger.csv
