"""Pool M2 result JSONs. usage: aggregate.py <gate: v3|strict> seed1.json seed2.json ... -> prints a table and writes results/m2_<gate>_pooled.json"""
import sys, json, numpy as np, collections
gate = sys.argv[1]; files = sys.argv[2:]; R = [json.load(open(f)) for f in files]
mags = sorted({float(k) for r in R for k in r['by_push_magnitude']})
def pooled(m):
    g = [r['by_push_magnitude'][str(m)] for r in R if str(m) in r['by_push_magnitude'] and r['by_push_magnitude'][str(m)]]
    s = lambda k: sum(x[k] for x in g)
    out = dict(n=s('n'), push_applied=s('push_applied'), fell_induced=s('fell_induced'), spontaneous_falls=s('spontaneous_falls'), stand_gate_pass=s('stand_gate_pass'), completed_walk_window=s('completed_walk_window'),
               refell_after_handover=s('refell_after_handover'), env_resets=s('env_resets'), episode_success=s('episode_success'))
    out['outcomes'] = dict(sum((collections.Counter(x['outcomes']) for x in g), collections.Counter()))
    pe = [(r['per_env']) for r in R]
    sel = lambda r, v: [i for i, mm in enumerate(r['per_env']['mag']) if abs(mm - m) < 1e-9]
    lat, gu, al = [], [], []
    for r in R:
        p = r['per_env']; T = r['timeline']
        for i in sel(r, m):
            if p['detect_step'][i] >= 0 and p['impulse_step'][i] >= 0 and p['state'][i] != 'walk' and (p['detect_step'][i] >= p['impulse_step'][i]): lat.append((p['detect_step'][i] - p['impulse_step'][i]) * T['dt'])
            if p['getup_s'][i] is not None: gu.append(p['getup_s'][i]); al.append(p['along_m'][i])
    f = lambda x: None if not x else dict(n=len(x), mean=round(float(np.mean(x)), 2), median=round(float(np.median(x)), 2), p90=round(float(np.percentile(x, 90)), 2))
    out.update(detection_latency_s=f(lat), getup_time_s=f(gu), along_heading_m=f(al)); return out
rows = {str(m): pooled(m) for m in mags}
tot = {k: sum(r[k] for r in rows.values()) for k in ['n', 'push_applied', 'fell_induced', 'spontaneous_falls', 'stand_gate_pass', 'completed_walk_window', 'refell_after_handover', 'env_resets', 'episode_success']}
print(f'gate={gate} seeds={[r["seed"] for r in R]} envs/seed={R[0]["num_envs"]}')
print('push m/s | n | fell | fell% | gate pass | walked 10 s | re-fell | resets | success/fell | latency s (mean) | get-up s (mean)')
for m, g in rows.items():
    if float(m) == 0: print(f'{m:>8} | {g["n"]} | {g["fell_induced"]+g["spontaneous_falls"]} (control: spontaneous falls) | - | - | - | - | {g["env_resets"]} | - | - | -'); continue
    print(f'{m:>8} | {g["n"]} | {g["fell_induced"]} | {100*g["fell_induced"]/g["n"]:.0f}% | {g["stand_gate_pass"]} | {g["completed_walk_window"]} | {g["refell_after_handover"]} | {g["env_resets"]} | {g["episode_success"]}/{g["fell_induced"]} | {g["detection_latency_s"]["mean"] if g["detection_latency_s"] else "-"} | {g["getup_time_s"]["mean"] if g["getup_time_s"] else "-"}')
pushed = {k: sum(rows[m][k] for m in rows if float(m) > 0) for k in ['n', 'fell_induced', 'stand_gate_pass', 'completed_walk_window', 'refell_after_handover', 'env_resets', 'episode_success']}
print('pushed envs total', pushed, f'-> success among induced falls: {pushed["episode_success"]}/{pushed["fell_induced"]} = {pushed["episode_success"]/max(pushed["fell_induced"],1):.3f}')
ctl = [r['control_no_push'] for r in R]; print('control (no push):', dict(n=sum(c['n'] for c in ctl), fell=sum(c['fell'] for c in ctl), still_walking_at_end=sum(c['still_walking_at_end'] for c in ctl), ref_along_m_mean=round(float(np.mean([c['walk_reference_along_heading_m_over_window']['mean'] for c in ctl if c.get('walk_reference_along_heading_m_over_window')])),2), ref_ge_3m=sum(c['reference_ge_3m'] for c in ctl), ref_ge_5m=sum(c['reference_ge_5m'] for c in ctl)))
nd = [r['no_stand_gate_diagnosis'] for r in R if r.get('no_stand_gate_diagnosis')]
if nd:
    cf = collections.Counter(); n = 0; reach = 0
    for d in nd:
        n += d['n']; reach += d['reached_strict_2s_stand']; cf.update(d['clause_failures'])
    print(f'timed-out recovery windows: {n}; reached a strict 2 s supported stand: {reach}; gate clauses failed (counts): {dict(cf)}')
mv = collections.defaultdict(lambda: [0, 0.0])
for r in R:
    for k, v in (r['movement_during_getup'] or {}).items():
        if v: mv[k][0] += v['n']; mv[k][1] += v['n'] * v['mean']
mvm = {k: round(b / a, 3) for k, (a, b) in mv.items() if a}
print('movement during get-up (mean over windows, n=%d):' % max((a for a, _ in mv.values()), default=0), mvm)
json.dump(dict(gate=gate, seeds=[r['seed'] for r in R], files=[f.split('/')[-1] for f in files], by_push_magnitude=rows, pushed_total=pushed, control=dict(n=sum(c['n'] for c in ctl), fell=sum(c['fell'] for c in ctl), still_walking_at_end=sum(c['still_walking_at_end'] for c in ctl)), timed_out_diagnosis_clause_failures=dict(cf) if nd else None, movement_mean=mvm), open(f'results/m2_{gate}_pooled.json', 'w'), indent=1)
