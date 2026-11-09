#!/bin/bash
# Reproducible gate test (CPU only, no GPU job). Each case must fail with the stated exit code; the good case must pass.
P=$(cd "$(dirname "$0")/.." && pwd); PY=$HOME/isaac-workspace/envs/lab-3.0.0/bin/python; I=$P/m0/ckpt_inventory.py; B=$P/ckpt/R9hg_it1100_BASELINE_frozen.pt; fail=0
t(){ want=$1; shift; out=$("$@" 2>/dev/null); rc=$?; if [ $rc = $want ]; then echo "PASS rc=$rc  $DESC"; else echo "FAIL rc=$rc (want $want) $DESC :: $out"; fail=1; fi; }
DESC="good baseline"                     ; t 0 $PY -I $I $B --sha256 665a89f58b3a --expect-obs 123 --expect-act 37
DESC="missing checkpoint file"           ; t 2 $PY -I $I $P/ckpt/does_not_exist.pt
DESC="wrong sha256"                      ; t 3 $PY -I $I $B --sha256 deadbeef0000
DESC="wrong expected obs dim"            ; t 6 $PY -I $I $B --expect-obs 124
DESC="not a checkpoint (text file)"      ; t 5 $PY -I $I $P/overrides.txt
DESC="run-dir discovery, none matches"   ; t 4 bash $P/m0/find_run.sh "$P/m0/nonexistent_root" nosuchrun
DESC="REGRESSION: empty logs root (the pilot driver bug: LAB= empty)" ; t 4 bash $P/m0/find_run.sh "/logs/rsl_rl/g1_recovery_r" humanlike_p1_s21
DESC="positive: real pilot run dir is found"  ; t 0 bash $P/m0/find_run.sh "$HOME/isaac-workspace/envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r" humanlike_p1_s21
exit $fail
