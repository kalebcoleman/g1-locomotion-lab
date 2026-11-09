#!/bin/bash
# usage: find_run.sh <logs_root> <run_name>  -> prints the single run dir named <timestamp>_<run_name>; exit 4 if none or more than one (never silent)
D=$(ls -d "$1"/*_"$2" 2>/dev/null); n=$(echo -n "$D" | grep -c .)
[ "$n" = 1 ] && { echo "$D"; exit 0; }
echo "run dir lookup failed: $n matches for '$1/*_$2'" >&2; exit 4
