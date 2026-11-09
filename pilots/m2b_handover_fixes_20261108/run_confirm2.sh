#!/bin/bash
# M2b confirmation of the readiness-gated best variant (evaluation only): repeats on seeds 2-4, fresh seeds 5-7.
P=$(cd "$(dirname "$0")" && pwd)
$P/run_variants.sh "2 3 4" "_r2" ready_upper50
$P/run_variants.sh "2 3 4" "_r3" ready_upper50
$P/run_variants.sh "5 6 7" "" ready_upper50
