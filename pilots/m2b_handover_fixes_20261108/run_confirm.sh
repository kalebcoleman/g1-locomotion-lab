#!/bin/bash
# M2b confirmation runs (evaluation only): variance repeats on seeds 2-4 and fresh seeds 5-7.
P=$(cd "$(dirname "$0")" && pwd)
$P/run_variants.sh "2 3 4" "_r2" base blendupper50
$P/run_variants.sh "2 3 4" "_r3" base blendupper50
$P/run_variants.sh "5 6 7" "" base blendupper50
