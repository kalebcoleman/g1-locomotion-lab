"""Goal 4 driver: short warm-started stages with a ramped early-rise stance penalty; every checkpoint is evaluated on DEV (selection) with the movement metrics; hard GPU-hour cap.
usage: driver_g4.py <plan.json>      plan = {name, init, seed, num_envs, iters, budget_hours_cap, stages: [{name, overrides: [...]}], target: {peak_spread_m, rate_vs_baseline}}
Rule (stated before running): a checkpoint is FEASIBLE if DEV v3 >= rate_vs_baseline * baseline DEV v3. The stage-best checkpoint is the feasible one with the smallest mean peak spread
(pelvis > 0.12 m), else the one with the highest DEV v3. The next stage starts from the stage-best. Stop when a feasible checkpoint has peak spread < target, when a stage has no
feasible checkpoint (ramp stopped), or when cumulative TRAINING wall time reaches the cap. HELDOUT is not looked at here."""
import sys, os, json, glob, subprocess, time, re, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from selection import pick_stage_best, target_met      # refactored after the runs to make the rule unit-testable; behaviour unchanged
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(HERE + '/../..'); REC = ROOT + '/recovery'
LABLOGS = os.path.expanduser('~/robotics-workspace/envs/lab-3.0.0-src/logs/rsl_rl/g1_recovery_r')
plan = json.load(open(sys.argv[1])); NAME = plan['name']; LOG = HERE + f'/results/{NAME}_driver.jsonl'
def log(**k):
    k['t'] = time.strftime('%H:%M:%S'); open(LOG, 'a').write(json.dumps(k) + '\n'); print(json.dumps(k), flush=True)
def hours_used():
    p = HERE + '/ledger_train.csv'
    return sum(float(l.split(',')[3]) for l in open(p) if l.strip()) / 3600.0 if os.path.exists(p) else 0.0
def evaluate(ck, label, sets):
    out = {}
    for sname, spath in sets.items():
        o = f'{HERE}/results/{label}_{sname}.json'
        s = time.time(); cmd = f'cd {REC} && ./scripts/run.sh eval_recovery.py --task G1-Recovery-R9hg-v0 --checkpoint {ck} --num_envs 256 --heldout {REC}/sets/{spath} --out {o} physics=newton_mjwarp'
        rc = subprocess.run(cmd, shell=True, stdout=open(f'{HERE}/logs/{label}_{sname}.log', 'w'), stderr=subprocess.STDOUT).returncode
        open(HERE + '/ledger_eval.csv', 'a').write(f'{label}_{sname},{s},{time.time()},{time.time()-s:.0f},rc={rc}\n')
        if rc != 0 or not os.path.exists(o): log(event='eval_failed', label=label, set=sname, rc=rc); return None
        r = json.load(open(o)); mv = r['movement']['all_robots']
        out[sname] = dict(v3=r['amendment3']['ground_only_flight']['v3_before_selfpen'], strict=r['strict_supported_stand']['successes'], peak_spread=mv['spread_peak_m']['mean'], peak_spread_gated=mv['spread_peak_gated_m']['mean'],
                          final_spread=mv['spread_final_m']['mean'], hard=r['plausibility']['hard_limit_step_frac_mean'], soft=r['plausibility']['pos_limit_violation_step_frac_mean'], by_cat=r['amendment3']['ground_only_flight']['by_category'])
    return out
SETS_DEV = {'dev': 'dev_v1.npz'}
base = plan.get('baseline_ckpt', plan['init']); res_base = evaluate(base, f'{NAME}_baseline', SETS_DEV); log(event='baseline_dev', baseline=os.path.basename(base), **res_base['dev'])      # the bar comes from the baseline, not from the start checkpoint
need = plan['target']['rate_vs_baseline'] * res_base['dev']['v3']; cur = plan['init']; best_overall = None
for st in plan['stages']:
    if hours_used() >= plan['budget_hours_cap']: log(event='budget_cap_reached', hours=hours_used()); break
    run = f"{NAME}_{st['name']}"; s = time.time()
    os.environ['SAVE_INTERVAL'] = str(plan.get('save_interval', 25)); EV = plan.get('eval_every', plan.get('save_interval', 25))
    cmd = [HERE + '/train_stage.sh', run, str(plan['num_envs']), str(st.get('iters', plan['iters'])), str(plan['seed']), cur] + st['overrides']
    log(event='train_start', stage=st['name'], init=os.path.basename(cur), overrides=st['overrides'], hours_before=round(hours_used(), 3))
    rc = subprocess.run(cmd, stdout=open(f'{HERE}/logs/{run}_train.log', 'w'), stderr=subprocess.STDOUT).returncode
    wall = time.time() - s; open(HERE + '/ledger_train.csv', 'a').write(f'{run},{s},{time.time()},{wall:.0f},rc={rc}\n')
    dirs = sorted(glob.glob(f'{LABLOGS}/*_{run}'))
    if rc != 0 or len(dirs) != 1: log(event='train_failed_or_no_run_dir', stage=st['name'], rc=rc, dirs=len(dirs)); break
    cks = sorted(glob.glob(dirs[0] + '/model_*.pt'), key=lambda p: int(re.findall(r'model_(\d+)', p)[0])); cks = [c for c in cks if int(re.findall(r'model_(\d+)', c)[0]) % EV == 0 or c == cks[-1]]
    cks = [c for c in cks if int(re.findall(r'model_(\d+)', c)[0]) > int(re.findall(r'model_(\d+)', cks[0])[0])] or cks
    rows = []
    for c in cks:
        it = int(re.findall(r'model_(\d+)', c)[0]); keep = f'{HERE}/ckpt/{run}_it{it}.pt'; shutil.copy(c, keep); os.chmod(keep, 0o444)
        e = evaluate(keep, f'{run}_it{it}', SETS_DEV)
        if e is None: continue
        rows.append(dict(ckpt=keep, it=it, **e['dev'])); log(event='ckpt_dev', stage=st['name'], it=it, **e['dev'])
    pick, feas = pick_stage_best(rows, need)
    log(event='stage_done', stage=st['name'], train_wall_s=round(wall), hours_total=round(hours_used(), 3), feasible=len(feas), picked=None if pick is None else dict(it=pick['it'], v3=pick['v3'], peak_spread=pick['peak_spread']))
    if pick is None: break
    if best_overall is None or (pick in feas and (best_overall['v3'] < need or pick['peak_spread'] < best_overall['peak_spread'])): best_overall = dict(pick, stage=st['name'], feasible=bool(feas))
    if not feas and plan.get('select', 'best') == 'best': log(event='ramp_stopped_no_feasible_checkpoint', stage=st['name'], need_v3=need); break
    cur = pick['ckpt'] if plan.get('select', 'best') == 'best' else rows[-1]['ckpt']      # 'last': the next stage continues the same trajectory (a gradual in-run ramp)
    if target_met(pick, feas, plan['target']['peak_spread_m']): log(event='target_met_on_dev', **pick); break      # only a FEASIBLE checkpoint (v3 bar met) can meet the target
log(event='finished', best=best_overall, hours_total=round(hours_used(), 3))
