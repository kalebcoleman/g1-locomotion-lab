#!/bin/bash
# usage: gpu.sh <label> <cmd...>   -- runs a GPU job and appends start/end epoch seconds to ledger.csv (aggregate GPU budget accounting)
P=$(cd "$(dirname "$0")" && pwd); L=$1; shift
s=$(date +%s); echo "START $L $(date +%T)" >> $P/logs/driver.log
"$@" > $P/logs/$L.log 2>&1; rc=$?
e=$(date +%s); echo "$L,$s,$e,$((e-s)),rc=$rc" >> $P/ledger.csv; echo "END $L rc=$rc $((e-s))s" >> $P/logs/driver.log
exit $rc   # propagate the job exit code (the pilot version swallowed it, so a failed job did not stop a driver)
