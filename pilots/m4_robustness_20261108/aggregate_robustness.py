"""Pool M4 results. usage: aggregate_robustness.py [--friction f1,f2,...] -> tables; reads results/m4_mu<f>_s<seed>.json; writes results/pooled.json"""
import sys, os, json, glob, numpy as np, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../recovery')
from g1_recovery import robustness_stats as RS
files = sorted(glob.glob(HERE + '/results/m4_mu*_s*.json')); by_mu = collections.defaultdict(list)
for f in files:
    r = json.load(open(f)); by_mu[r['friction']].append(r)
def counts(rs, sel=lambda m, d: True):
    """falls per magnitude AMONG ROBOTS STILL STANDING AT THE PUSH (robots that fell before the push never felt it; they are reported separately)."""
    c = collections.defaultdict(lambda: [0, 0])
    for r in rs:
        p = r['per_env']; PRE = int(round(r['pre_s'] / 0.02))
        for m, d, fl, fs in zip(p['mag'], p['dir_deg'], p['fell'], p['fall_step']):
            if fl and 0 <= fs < PRE: continue
            if sel(m, d): c[m][0] += fl; c[m][1] += 1
    mags = sorted(c); return mags, [c[m][0] for m in mags], [c[m][1] for m in mags]
out = {}
print('friction | seeds | robots | walked the first 3 s without falling [95% CI] | standing at the push, per magnitude | falls with NO push after those 3 s | M50 interpolated | M50 logistic [95% bootstrap CI]')
for mu in sorted(by_mu, reverse=True):
    rs = by_mu[mu]; mags, k, n = counts(rs); m50i = RS.m50_interp(mags, k, n); m50l = RS.m50_logistic(mags, k, n); lo, hi, nv = RS.m50_bootstrap_ci(mags, k, n)
    k0, n0 = (k[0], n[0]) if mags and mags[0] == 0 else (0, 0); ci0 = RS.wilson_ci(k0, n0); pre = sum(r['fell_before_push'] for r in rs); tot = sum(r['num_envs'] for r in rs); cip = RS.wilson_ci(tot - pre, tot)
    out[mu] = dict(seeds=[r['seed'] for r in rs], mags=mags, falls=k, n=n, m50_interp=m50i, m50_logistic=m50l, m50_ci=[lo, hi], no_push_falls=[k0, n0, [round(v, 3) for v in ci0]], fell_before_push=pre, robots=tot, pre_push_survival_ci=[round(v, 3) for v in cip])
    fmt = lambda v: '-' if v is None else f'{v:.2f}'
    print(f"{mu} | {[r['seed'] for r in rs]} | {tot} | {tot-pre}/{tot} ({100*(tot-pre)/tot:.0f}% [{100*cip[0]:.0f},{100*cip[1]:.0f}]) | {min(n) if n else 0}-{max(n) if n else 0} | {k0}/{n0} | {fmt(m50i)} | {fmt(m50l)} [{fmt(lo)}, {fmt(hi)}]")
mu0 = max(by_mu)   # default training floor friction (1.0)
print(f'\nSurvival vs push magnitude, floor friction {mu0} (pooled over directions and seeds): magnitude | survived/total | survival [95% CI]')
d = out[mu0]
for m, k, n in zip(d['mags'], d['falls'], d['n']):
    lo, hi = RS.wilson_ci(n - k, n); print(f"  {m:g} m/s | {n-k}/{n} | {100*(n-k)/n:.0f}% [{100*lo:.0f}, {100*hi:.0f}]")
print(f'\nBy push direction (floor friction {mu0}; 0 = forward, 90 = left, 180 = backward, 270 = right): M50 (interpolated, logistic) and falls at 2.0 m/s')
dirs = sorted({x for r in by_mu[mu0] for x in r['per_env']['dir_deg']}); dd = {}
for dg in dirs:
    mags, k, n = counts(by_mu[mu0], lambda m, d_, dg=dg: abs(d_ - dg) < 1e-6 or m == 0)   # the no-push control is shared by every direction
    i2 = mags.index(2.0) if 2.0 in mags else None; mi = RS.m50_interp(mags, k, n); ml = RS.m50_logistic(mags, k, n); dd[dg] = dict(m50_interp=mi, m50_logistic=ml, falls_at_2=None if i2 is None else [k[i2], n[i2]])
    print(f"  {dg:g} deg | M50 {('-' if mi is None else f'{mi:.2f}')} / {('-' if ml is None else f'{ml:.2f}')} | falls at 2.0 m/s: {dd[dg]['falls_at_2'][0]}/{dd[dg]['falls_at_2'][1]}")
json.dump(dict(by_friction=out, by_direction_at_default_friction=dd), open(HERE + '/results/pooled.json', 'w'), indent=1)
