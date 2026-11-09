"""Summarise final_eval results: usage summarize_final.py <tag>"""
import json, sys, glob, numpy as np
tag = sys.argv[1]; D = __file__.rsplit('/', 1)[0] + '/results/'
def row(f):
    r = json.load(open(f)); mv = r['movement']['all_robots']; g = r['amendment3']['ground_only_flight']
    return dict(v3=g['v3_before_selfpen'], strict=r['strict_supported_stand']['successes'], cat=g['by_category'], peak=mv['spread_peak_m']['mean'], peak_g=mv['spread_peak_gated_m']['mean'], final=mv['spread_final_m']['mean'], wide=mv['spread_wide_frac']['mean'],
                hard=r['plausibility']['hard_limit_step_frac_mean'], soft=r['plausibility']['pos_limit_violation_step_frac_mean'], acc=mv['joint_acc_rms']['mean'], arate=mv['action_rate_mean']['mean'], hr=mv['hip_roll_peak_abs']['mean'], hy=mv['hip_yaw_peak_abs']['mean'])
for who in ('base', tag):
    rs = [row(f) for f in sorted(glob.glob(D + f'final_{who}_heldout_r*.json'))]
    if not rs: continue
    print(f'{who} HELDOUT x{len(rs)}: v3', [r['v3'] for r in rs], 'strict', [r['strict'] for r in rs], 'peak spread (pelvis>0.12 m) %.3f' % np.mean([r['peak'] for r in rs]), 'peak(>0.40 m) %.3f' % np.mean([r['peak_g'] for r in rs]), 'final %.3f' % np.mean([r['final'] for r in rs]),
          'wide-frac %.2f' % np.mean([r['wide'] for r in rs]), 'hard %.4f soft %.3f' % (np.mean([r['hard'] for r in rs]), np.mean([r['soft'] for r in rs])), 'acc rms %.0f action rate %.3f' % (np.mean([r['acc'] for r in rs]), np.mean([r['arate'] for r in rs])), 'hip peak roll/yaw %.2f/%.2f' % (np.mean([r['hr'] for r in rs]), np.mean([r['hy'] for r in rs])))
    print('   by direction (side_a, prone, side_b, supine):', [[r['cat'][k] for k in ('side_a', 'prone', 'side_b', 'supine')] for r in rs])
for who in ('base', tag):
    try: r = row(D + f'final_{who}_dev.json'); print(f'{who} DEV: v3 {r["v3"]} strict {r["strict"]} peak {r["peak"]:.3f} final {r["final"]:.3f} hard {r["hard"]:.4f} soft {r["soft"]:.3f}')
    except Exception: pass
