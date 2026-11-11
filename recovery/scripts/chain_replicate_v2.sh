#!/bin/bash
# Exact replication of the seed-3 chain that produced R9hg it1100, on another seed. Hand-over iterations are fixed in advance
# (no DEV look between stages): R7h scratch -> model_650 -> R9h -> model_900 -> R9hc -> model_1000 -> R9hg -> model_1100.
# usage: chain_replicate.sh <seed>
S=$1; LAB=~/isaac-workspace/envs/lab-3.0.0-src; PY=~/isaac-workspace/envs/lab-3.0.0/bin/python; L=$LAB/logs/rsl_rl/g1_recovery_r; ROOT=~/isaac-workspace/projects/g1-locomotion-lab-integrated
export PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$ROOT/recovery${PYTHONPATH:+:$PYTHONPATH}
stage(){ # task max_its want_iteration [resume_ckpt]
  local before=$(ls -d $L/20* | sort | tail -1); local extra=""; [ -n "$4" ] && extra="--checkpoint $4 --no-resume_optimizer_state"
  echo "STAGE $1 seed $S start $(date +%T) resume ${4:-none}"
  (cd $LAB && $PY scripts/reinforcement_learning/train.py --task $1 --num_envs 16384 --rl_library rsl_rl --max_iterations $2 --seed $S --run_name chain_s${S}_$1 --viz none --external_callback=g1_recovery.register.register $extra > $ROOT/archive/runs/chain_s${S}_$1.log 2>&1)
  local d=$(ls -d $L/*_chain_s${S}_$1 2>/dev/null | sort | tail -1)   # unique folder per seed and stage (two seeds started in the same second shared one folder before this fix)
  CK=$L/init/chain_s${S}_$1_it$3.pt; cp $d/model_$3.pt $CK || { echo "MISSING $d/model_$3.pt"; exit 1; }
  echo "STAGE_DONE $1 seed $S dir $d ckpt $CK $(sha256sum $CK | cut -c1-16)"
}
stage G1-Recovery-R7h-v0 700 650
stage G1-Recovery-R9h-v0 300 900 $CK
stage G1-Recovery-R9hc-v0 150 1000 $CK
stage G1-Recovery-R9hg-v0 150 1100 $CK
echo CHAIN_DONE seed $S
