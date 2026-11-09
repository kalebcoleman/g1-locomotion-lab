"""Pool M3 terrain results over seeds. usage: aggregate_terrain.py r1.json r2.json ... -> prints tables and writes pooled.json"""
import sys, json, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'recovery'))
from g1_recovery import robustness_stats as RS
R = [json.load(open(f)) for f in sys.argv[1:]]; seeds = [r['seed'] for r in R]; desc = R[0]['level_descriptors']; H = R[0]['horizon_s']
types = list(R[0]['by_type_level'].keys()); out = {}
def pool(t, l):
    c = [r['by_type_level'][t][str(l)] for r in R]; k = sum(x['survived'] for x in c); n = sum(x['n'] for x in c)
    # distance/speed over survivors, weighted by survivors per seed
    w = [x['survived'] for x in c if x['along_m_survivors_mean'] is not None]; d = [x['along_m_survivors_mean'] for x in c if x['along_m_survivors_mean'] is not None]
    dm = float(np.average(d, weights=w)) if w and sum(w) else None
    ps = [round(x['rate'], 2) for x in c]
    return dict(k=k, n=n, rate=k / n, ci=[round(v, 3) for v in RS.wilson_ci(k, n)], dist=dm, speed=None if dm is None else dm / H, per_seed=ps)
lab = lambda t, l: (f"slope {desc['slope_deg_by_level'][l]:.0f} deg" if t.startswith('slope') else f"step {desc['step_height_m_by_level'][l]*100:.1f} cm" if t.startswith('steps') else f"noise {desc['rough_noise_amplitude_m'][int(t.split('_')[1])]*100:.0f} cm" if t.startswith('rough') else 'flat')
print(f'seeds {seeds}; {R[0]["reps_per_cell"]} robots per cell per seed; horizon {H:.0f} s; command {R[0]["fwd_cmd"]} m/s; blind walking (no height scan)')
print('terrain | level (descriptor at level centre) | survived 10 s | rate [95% CI] | distance of survivors (m) | speed (m/s) | per-seed rate')
for t in types:
    lv = sorted(int(x) for x in R[0]['by_type_level'][t]); out[t] = {}
    for l in lv:
        p = pool(t, l); out[t][l] = p
        print(f"{t} | {l} ({lab(t, l)}) | {p['k']}/{p['n']} | {100*p['rate']:.0f}% [{100*p['ci'][0]:.0f}, {100*p['ci'][1]:.0f}] | {'-' if p['dist'] is None else round(p['dist'],2)} | {'-' if p['speed'] is None else round(p['speed'],2)} | {p['per_seed']}")
print('\nhardest level with >= 80% survival (pooled; every easier level must also pass):')
hard = {}
for t in types:
    if t.startswith('rough_') or t == 'flat': continue
    lv = sorted(out[t]); h = RS.hardest_level_with_rate(lv, {l: (out[t][l]['k'], out[t][l]['n']) for l in lv}, 0.8)
    ph = []
    for i, r in enumerate(R):
        c = {l: (r['by_type_level'][t][str(l)]['survived'], r['by_type_level'][t][str(l)]['n']) for l in lv}; ph.append(RS.hardest_level_with_rate(lv, c, 0.8))
    hard[t] = dict(level=h, descriptor=None if h is None else lab(t, h), per_seed=ph); print(f"  {t}: level {h} ({hard[t]['descriptor']}); per seed {ph}")
rk = sorted(int(t.split('_')[1]) for t in types if t.startswith('rough_')); rl = {k: (out[f'rough_{k}'][0]['k'], out[f'rough_{k}'][0]['n']) for k in rk}
h = RS.hardest_level_with_rate(rk, rl, 0.8); hard['rough'] = dict(level=h, descriptor=None if h is None else lab(f'rough_{h}', 0)); print(f"  rough ground: level {h} ({hard['rough']['descriptor']})")
gt = [t for t in types if t in ('slope_down', 'slope_up', 'steps_down', 'steps_up')]; lv = sorted(out[gt[0]])
pl = {l: (sum(out[t][l]['k'] for t in gt), sum(out[t][l]['n'] for t in gt)) for l in lv}; hp = RS.hardest_level_with_rate(lv, pl, 0.8)
print('  pooled over the four graded types (slope_down, slope_up, steps_down, steps_up), by level index:', {l: f'{k}/{n} ({100*k/n:.0f}%)' for l, (k, n) in pl.items()}); print(f'  -> hardest pooled level with >= 80%: {hp}'); hard['pooled_graded'] = dict(level=hp, by_level={str(l): pl[l] for l in lv})
json.dump(dict(seeds=seeds, by_type_level=out, hardest_level_80=hard, level_descriptors=desc), open(os.path.dirname(os.path.abspath(__file__)) + '/results/pooled.json', 'w'), indent=1)
