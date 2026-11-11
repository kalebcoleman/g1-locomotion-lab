"""Compare M2c variants (M2b hand-over protocol rerun with the recC get-up baseline).
usage: aggregate_m2c.py [--detail] variant1 variant2 ...
Reads results/m2c_<variant>_<run-tag>_s<seed>.json, where run-tag r1/r2/r3 are the three independent repeats on seeds 2,3,4 and run-tag fresh is the single pass on seeds 5,6,7.
Columns follow M2/M2b: induced falls, hand-overs (the switch happened), re-fell after the switch, mean time-to-re-fall, walked 10 s, full success (M2 rule: no re-fall,
>= 3.0 m along the heading and >= 0.5 m/s), success rate of induced falls, per-seed full successes, hand-overs whose OFFICIAL v3 gate was true at the switch
(readiness-trigger variants can hand over without it) and full successes among those. Added here: the pooled rate's 95% Wilson score interval (recovery/g1_recovery/robustness_stats.py)."""
import sys, os, json, importlib.util
import numpy as np

P = os.path.dirname(os.path.abspath(__file__)); D = P + '/results/'
ROOT = os.path.dirname(os.path.dirname(P))
_s = importlib.util.spec_from_file_location('robustness_stats', os.path.join(ROOT, 'recovery', 'g1_recovery', 'robustness_stats.py'))
RS = importlib.util.module_from_spec(_s); _s.loader.exec_module(RS)

BLOCKS = [('seeds 2-4, run 1', 'r1', [2, 3, 4]), ('seeds 2-4, run 2', 'r2', [2, 3, 4]), ('seeds 2-4, run 3', 'r3', [2, 3, 4]), ('seeds 2-4, pooled', ('r1', 'r2', 'r3'), [2, 3, 4]),
          ('fresh seeds 5-7', 'fresh', [5, 6, 7])]
detail = '--detail' in sys.argv; dist = '--dist' in sys.argv; args = [a for a in sys.argv[1:] if not a.startswith('--')]

def load(tag, runs, seeds):
    out = []
    for run in runs:
        for s in seeds:
            f = D + f'm2c_{tag}_{run}_s{s}.json'
            if not os.path.exists(f): print('MISSING', f); return None
            out.append(json.load(open(f)))
    return out

def row(tag, label, R):
    O = [r['overall'] for r in R]; S = lambda k: sum(o[k] for o in O)
    tt = [v for r in R for v in r['per_env']['time_to_refall_s'] if v is not None]
    k, n = S('episode_success'), S('fell_induced'); lo, hi = RS.wilson_ci(k, n)
    return dict(variant=tag, block=label, runs=len(R), fell=n, ho=S('stand_gate_pass'), refell=S('refell_after_handover'),
                ttr=(round(float(np.mean(tt)), 2) if tt else None), walked=S('completed_walk_window'), ge3=S('walked_10s_and_ge3m'),
                succ=k, rate=k / max(n, 1), wilson=[round(100 * lo, 1), round(100 * hi, 1)], per_seed=[o['episode_success'] for o in O],
                per_seed_refell=[o['refell_after_handover'] for o in O], v3ho=S('handover_with_v3_true'), succ_v3=S('success_with_v3_true'),
                ctl=[r['control_no_push']['still_walking_at_end'] for r in R], ctl_n=[r['control_no_push']['n'] for r in R],
                along=[r['overall']['along_heading_m']['mean'] for r in R if r['overall']['along_heading_m']],
                walk=[r['overall']['along_heading_m']['mean'] for r in R if r['overall']['along_heading_m']])

print(f'M2c: recC get-up baseline, identical protocol/pushes/walking policy to M2b; 128 envs/seed (32 no-push controls + 96 pushed per seed); numbers summed per row')
print('variant | block | runs | induced falls | hand-overs | re-fell | mean t-to-refall s | walked 10 s | >=3 m | full success | rate (95% Wilson CI) | per-seed success | per-seed re-fell | v3 true at switch | success with v3 true | control walking')
for tag in args:
    rows = []
    for label, runs, seeds in BLOCKS:
        runs = runs if isinstance(runs, tuple) else (runs,)
        R = load(tag, runs, seeds)
        if R is None: continue
        rows.append(row(tag, label, R))
        x = rows[-1]
        print(f"{tag} | {label} | {x['runs']} | {x['fell']} | {x['ho']} | {x['refell']} | {x['ttr']} | {x['walked']} | {x['ge3']} | {x['succ']} | "
              f"{100*x['rate']:.1f}% ({x['wilson'][0]}-{x['wilson'][1]}%) | {x['per_seed']} | {x['per_seed_refell']} | {x['v3ho']} | {x['succ_v3']} | {sum(x['ctl'])}/{sum(x['ctl_n'])}")
    if dist:
        print(f'\n## {tag}: post-hand-over walking distance (mean along-heading m per completed window, mean over runs) and control reference')
        for label, runs, seeds in BLOCKS:
            runs = runs if isinstance(runs, tuple) else (runs,)
            R = load(tag, runs, seeds)
            if R is None: continue
            a = [r['overall']['along_heading_m']['mean'] for r in R if r['overall']['along_heading_m']]
            c = [r['control_no_push']['walk_reference_along_heading_m_over_window']['mean'] for r in R if r['control_no_push']['walk_reference_along_heading_m_over_window']]
            print(f"{tag} | {label} | along m: {round(float(np.mean(a)), 2) if a else None} | control {round(float(np.mean(c)), 2) if c else None}")
    if detail:
        for label, runs, seeds in BLOCKS:
            if 'pooled' in label: continue
            runs = runs if isinstance(runs, tuple) else (runs,)
            R = load(tag, runs, seeds)
            if R is None: continue
            print(f'\n## {tag} / {label} (by push magnitude, summed over seeds in the block)')
            print('push m/s | episodes | fell | hand-over | re-fell | walked 10 s | full success')
            for m in R[0]['by_push_magnitude']:
                g = [r['by_push_magnitude'][m] for r in R]; s = lambda k: sum(x[k] for x in g)
                if float(m) == 0:
                    print(f"{m} | {s('n')} | control | - | - | {sum(r['control_no_push']['still_walking_at_end'] for r in R)} still walking | -"); continue
                print(f"{m} | {s('n')} | {s('fell_induced')} | {s('stand_gate_pass')} | {s('refell_after_handover')} | {s('completed_walk_window')} | {s('episode_success')}")
