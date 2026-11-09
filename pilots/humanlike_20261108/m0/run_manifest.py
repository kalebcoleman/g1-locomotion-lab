"""Run provenance + resume-mode proof from runtime artifacts (CPU only, read-only). usage: run_manifest.py <run_dir> <start_ckpt> <out.json> [--overrides overrides.txt]
Resume-mode rule: a checkpoint saves actor, critic, optimizer (Adam step counter) and iter. After training N iterations from a start checkpoint, compare:
 iter restored      -> first saved model_<k> has k == start iter (+ n)
 optimizer resumed  -> Adam step in a later checkpoint > Adam step in the start checkpoint; reset -> small step counter (about 20 per iteration)"""
import sys, os, json, hashlib, subprocess, glob, re, argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('start'); ap.add_argument('out'); ap.add_argument('--overrides'); a = ap.parse_args()
sh = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
ld = lambda p: torch.load(p, map_location='cpu', weights_only=True)
root = os.path.expanduser('~/isaac-workspace/projects/g1-locomotion-lab-integrated')
g = lambda *c: subprocess.run(['git', '-C', root, *c], capture_output=True, text=True).stdout.strip()
env = open(a.run + '/params/env.yaml').read(); ag = open(a.run + '/params/agent.yaml').read()
cks = sorted(glob.glob(a.run + '/model_*.pt'), key=lambda p: int(re.findall(r'model_(\d+)', p)[0]))
s = ld(a.start); first, last = ld(cks[0]), ld(cks[-1]); step = lambda c: float(c['optimizer_state_dict']['state'][0]['step'])
ov = {}
if a.overrides:
    for l in open(a.overrides).read().split():
        k, v = l.split('='); ov[k] = v
def saved(k):
    if k.startswith('env.rewards.'):
        t = k.split('.')[2]; m = re.search(rf'^  {t}:\n(?:    .*\n)*?    weight: (\S+)', env, re.M); return m.group(1) if m else None
    if k.startswith('agent.'): m = re.search(rf'^{k.split(".")[1]}: (\S+)', ag, re.M); return m.group(1) if m else None
out = dict(run_dir=os.path.basename(a.run), git_head=g('rev-parse', 'HEAD'), git_branch=g('branch', '--show-current'), git_untracked=len(g('status', '--short').splitlines()), git_tracked_modified=len([l for l in g('status', '--short').splitlines() if not l.startswith('??')]),
    env_yaml_sha256=sh(a.run + '/params/env.yaml'), agent_yaml_sha256=sh(a.run + '/params/agent.yaml'), start_ckpt_sha256=sh(a.start), checkpoints=[os.path.basename(c) for c in cks],
    overrides_requested=ov, overrides_in_saved_config={k: saved(k) for k in ov},
    start=dict(iter=int(s['iter']), adam_step=step(s)), first_saved=dict(name=os.path.basename(cks[0]), iter=int(first['iter']), adam_step=step(first)), last_saved=dict(name=os.path.basename(cks[-1]), iter=int(last['iter']), adam_step=step(last)),
    iteration_counter_restored=int(first['iter']) == int(s['iter']), optimizer_state_resumed=step(last) > step(s),
    checkpoint_contents=list(s.keys()), obs_normalizer_in_checkpoint=False, curriculum_state_in_checkpoint=False,
    notes='infos is None in checkpoints; the pose curriculum level (env._rec_level) is therefore not saved and restarts per config. Actor includes the learned action std (distribution.std_param). No observation normalizer keys exist.')
out['resume_mode'] = ('full resume' if out['iteration_counter_restored'] and out['optimizer_state_resumed'] else 'weights + iteration counter restored, optimizer RESET (warm start)' if out['iteration_counter_restored'] else 'weights only')
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps({k: out[k] for k in ['resume_mode', 'iteration_counter_restored', 'optimizer_state_resumed', 'start', 'first_saved', 'last_saved', 'overrides_in_saved_config', 'git_untracked', 'git_tracked_modified']}, indent=1))
