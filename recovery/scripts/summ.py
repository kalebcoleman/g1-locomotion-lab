#!/usr/bin/env python3
"""Compact table of v2 evaluation JSONs: summ.py results/a.json results/b.json ..."""
import json, sys, os
print(f"{'run':26s} {'legacy':>6s} {'strict':>6s} {'v2plaus':>7s} {'v2.1':>5s} {'hard%':>6s} {'hardX':>5s} {'ever_supp':>9s} {'hold_s':>6s} {'flight':>6s} {'pkvz':>5s} {'poslim':>6s} {'velviol':>7s} {'rec_t':>5s} {'assist_s':>8s}  plaus by noise | causes")
for f in sys.argv[1:]:
    d = json.load(open(f)); s = d["strict_supported_stand"]; p = d["plausibility"]; a = d.get("assist") or {}
    bd = {k: round(v.get("plausible_v21_rate", v["plausible_v2_rate"]), 3) for k, v in d.get("by_difficulty_limb_noise_rad", {}).items()}
    c = d["failure_causes"]
    print(f"{os.path.basename(f)[:-5]:26s} {d['successes']:6d} {s['successes']:6d} {p['plausible_v2_successes']:7d} {p.get('plausible_v21_successes',-1):5d} {100*p.get('hard_limit_step_frac_mean',-0.01):6.1f} {p.get('hard_excess_rad_mean_of_env_max',-1):5.2f} {s['envs_ever_supported_stand']:9d} {s['longest_stand_s_mean']:6.2f} {p['flight_step_frac_mean']:6.3f} {p['peak_pelvis_vz_mean']:5.2f} {p['pos_limit_violation_step_frac_mean']:6.2f} {p['vel_limit_violation_step_frac_mean']:7.3f} {str(round(s['recovery_time_s_mean'],2)) if s['recovery_time_s_mean'] else '-':>5s} {a.get('force_seconds_mean',0):8.2f}  {bd} | nr={c['never_rose']} rnu={c['rose_not_upright']} stf={c['stood_then_fell']} late={c['late_hold_incomplete']}")

print()
from itertools import chain
names = None
for f in sys.argv[1:]:
    d = json.load(open(f)); pr = d.get("postures_reached")
    if not pr: continue
    if names is None:
        names = list(pr); print(f"{'postures reached (envs)':26s} " + " ".join(f"{n[:13]:>13s}" for n in names))
    print(f"{os.path.basename(f)[:-5]:26s} " + " ".join(f"{pr[n]['envs']:13d}" for n in names))
