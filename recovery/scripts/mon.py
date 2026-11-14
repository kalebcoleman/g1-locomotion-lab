#!/usr/bin/env python3
"""Status of running/finished MCP training jobs from their stdout logs: mon.py [jobdir-suffix ...]; default: today's G1-Recovery jobs."""
import glob, os, re, sys, numpy as np
J = os.path.join(os.environ.get("TWINSMITH_JOBS_DIR") or os.path.expanduser("~/.local/share/twinsmith/jobs"), "")
dirs = sorted(d for d in glob.glob(J + "*G1-Recovery*") if os.path.getsize(d + "/stdout.log") > 20000)
if len(sys.argv) > 1: dirs = [d for d in dirs if any(a in d for a in sys.argv[1:])]
tot = 0
for d in dirs:
    t = open(d + "/stdout.log", errors="ignore").read().replace("\x00", "")
    m = re.findall(r"Learning iteration (\d+)/(\d+)", t)
    if not m: continue
    done = os.path.exists(d + "/exit_code")
    g = lambda k: (re.findall(k + r": ([\d.\-]+)", t) or ["-"])[-1]
    sps = [float(x) for x in re.findall(r"Steps per second: (\d+)", t)][-20:]
    a = np.mean(sps) if sps else 0
    if not done: tot += a
    print(f"{os.path.basename(d)[9:]:32s} {'DONE' if done else 'run '} it {m[-1][0]}/{m[-1][1]} sps {a:7.0f} R {g('Mean reward'):>7s} | height {g('Episode_Reward/height'):>7s} up {g('Episode_Reward/upright'):>7s} stand {g('Episode_Reward/stand'):>7s} | lvl {g('Curriculum/levels'):>6s} assist {g('Curriculum/assist_level'):>6s} | lying: supp {g('Curriculum/lying_ever_supported'):>6s} hold {g('Curriculum/lying_strict_hold'):>6s} maxh {g('Curriculum/lying_max_h'):>6s} | std {g('Mean action std'):>5s}")
print("COMBINED running throughput:", round(tot), "trans/s")
