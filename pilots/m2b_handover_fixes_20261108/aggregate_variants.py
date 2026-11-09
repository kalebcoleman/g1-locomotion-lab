"""Compare M2b variants. usage: aggregate_variants.py [--detail] [--seeds 2,3,4] variantTag1 variantTag2 ...   (reads results/m2b_<tag>_s<seed>.json)
Columns follow M2: induced falls, hand-overs (the switch happened), re-fell after hand-over, mean time-to-re-fall, walked 10 s, full success (M2 rule: no re-fall, >= 3.0 m, >= 0.5 m/s),
success rate of induced falls, per-seed full successes, hand-overs whose OFFICIAL v3 gate was true at the switch (readiness-trigger variants can hand over without it) and full successes among those."""
import sys, json, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + '/results/'
detail = '--detail' in sys.argv; args = [a for a in sys.argv[1:] if a != '--detail']
seeds = [2, 3, 4]
if args and args[0] == '--seeds': seeds = [int(x) for x in args[1].split(',')]; args = args[2:]
rows = []
def load(tag): return [json.load(open(D + f'm2b_{tag}_s{s}.json')) for s in seeds if os.path.exists(D + f'm2b_{tag}_s{s}.json')]
for tag in args:
    R = load(tag)
    if len(R) != len(seeds): print('missing', tag, len(R)); continue
    O = [r['overall'] for r in R]; S = lambda k: sum(o[k] for o in O)
    ttr = [x for r in R for x in [None] if False]
    tt = []; 
    for r in R:
        p = r['per_env']; tt += [v for v in p['time_to_refall_s'] if v is not None]
    rows.append(dict(tag=tag, fell=S('fell_induced'), ho=S('stand_gate_pass'), refell=S('refell_after_handover'), ttr=(round(float(np.mean(tt)), 2) if tt else None), walked=S('completed_walk_window'), ge3=S('walked_10s_and_ge3m'),
                     succ=S('episode_success'), rate=S('episode_success') / max(S('fell_induced'), 1), per_seed=[o['episode_success'] for o in O], per_seed_refell=[o['refell_after_handover'] for o in O], per_seed_ho=[o['stand_gate_pass'] for o in O],
                     v3ho=S('handover_with_v3_true'), succ_v3=S('success_with_v3_true'), cancels=S('settle_cancels'), ctl=[r['control_no_push']['still_walking_at_end'] for r in R], ctl_n=[r['control_no_push']['n'] for r in R]))
print(f'seeds {seeds}, 128 envs/seed (96 controls + 288 pushed per seed pair-sum); numbers summed over seeds')
print('variant | induced falls | hand-overs | re-fell | mean t-to-refall s | walked 10 s | >=3 m | full success | rate | per-seed success | per-seed re-fell | v3 true at switch | success with v3 true | control walking')
for x in rows:
    print(f"{x['tag']} | {x['fell']} | {x['ho']} | {x['refell']} | {x['ttr']} | {x['walked']} | {x['ge3']} | {x['succ']} | {100*x['rate']:.1f}% | {x['per_seed']} | {x['per_seed_refell']} | {x['v3ho']} | {x['succ_v3']} | {sum(x['ctl'])}/{sum(x['ctl_n'])}")
if detail:
    for tag in args:
        R = load(tag)
        if len(R) != len(seeds): continue
        print(f'\n## {tag} (by push magnitude, summed over seeds)')
        print('push m/s | episodes | fell | hand-over | re-fell | walked 10 s | full success')
        for m in R[0]['by_push_magnitude']:
            g = [r['by_push_magnitude'][m] for r in R]; s = lambda k: sum(x[k] for x in g)
            if float(m) == 0: print(f"{m} | {s('n')} | control | - | - | {sum(r['control_no_push']['still_walking_at_end'] for r in R)} still walking | -"); continue
            print(f"{m} | {s('n')} | {s('fell_induced')} | {s('stand_gate_pass')} | {s('refell_after_handover')} | {s('completed_walk_window')} | {s('episode_success')}")
