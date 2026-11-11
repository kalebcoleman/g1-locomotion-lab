#!/bin/bash
# M2b videos: the same 16 robots (seed 3, pushes 0/2.5/3.5/4.5 m/s), uncut, baseline vs the two best variants.
P=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$P/../.." && pwd); CK=$ROOT/checkpoints
# recovery policy: the tracked copy of the checkpoint the recorded results used. Override RECOVERY_CKPT to rerun with another one.
RECOVERY_CKPT=${RECOVERY_CKPT:-$ROOT/recovery/checkpoints/R9hg_it1100_previous_baseline.pt}; cd $ROOT/recovery; mkdir -p $P/media
run(){ L=$1; shift; ./scripts/run.sh eval_fall_handover.py --task G1-Recovery-R9hg-v0 --checkpoint $RECOVERY_CKPT --walk_checkpoint $CK/walking_newton_it1499.pt --num_envs 16 --seed 3 --levels 0,2.5,3.5,4.5 --record_dir $P/media --label $L --out $P/results/video_$L.json "$@" physics=newton_mjwarp > $P/logs/video_$L.log 2>&1; }
run m2b_base_seed3_16envs
run m2b_blendupper50_seed3_16envs --settle 50 --blend 50 --blend_joints upper
run m2b_ready_upper50_seed3_16envs --trigger ready --settle 50 --blend 50 --blend_joints upper
echo VIDEOS_DONE >> $P/ledger.csv
