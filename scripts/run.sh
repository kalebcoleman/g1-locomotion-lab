#!/bin/bash
# helper: run a g1-recovery python script from the Isaac Lab checkout. usage: run.sh <script.py> args...
export OMNI_KIT_ACCEPT_EULA=YES
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$ROOT:$PYTHONPATH
cd $HOME/isaac-workspace/envs/lab-3.0.0-src
exec ../lab-3.0.0/bin/python "$ROOT/scripts/$1" "${@:2}"
