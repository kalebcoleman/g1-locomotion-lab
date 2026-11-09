"""CPU-only checkpoint inventory (rsl_rl). usage: ckpt_inventory.py <ckpt.pt> [--sha256 H] [--expect-obs 123 --expect-act 37]
Exit codes: 0 ok; 2 file missing; 3 sha256 mismatch; 5 unreadable/unexpected structure; 6 shape mismatch. Prints one JSON line on stdout."""
import sys, os, json, hashlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('--sha256'); ap.add_argument('--expect-obs', type=int); ap.add_argument('--expect-act', type=int); a = ap.parse_args()
def die(code, msg): print(json.dumps(dict(ok=False, error=msg, ckpt=os.path.basename(a.ckpt)))); sys.exit(code)
if not os.path.isfile(a.ckpt): die(2, 'checkpoint file missing')
h = hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest()
if a.sha256 and not h.startswith(a.sha256): die(3, f'sha256 mismatch: got {h[:16]}, expected prefix {a.sha256}')
import torch
try: c = torch.load(a.ckpt, map_location='cpu', weights_only=True)
except Exception as e: die(5, f'unreadable: {str(e)[:120]}')
need = ['actor_state_dict', 'critic_state_dict', 'optimizer_state_dict', 'iter']
miss = [k for k in need if k not in c]
if miss: die(5, f'missing keys {miss}; found {list(c)}')
am = c['actor_state_dict']; obs = int(am['mlp.0.weight'].shape[1]); act = int(am['mlp.6.weight'].shape[0])
if (a.expect_obs and obs != a.expect_obs) or (a.expect_act and act != a.expect_act): die(6, f'shape mismatch: obs {obs} act {act}')
os_ = c['optimizer_state_dict']; adam_step = float(os_['state'][0]['step']) if os_.get('state') else None
print(json.dumps(dict(ok=True, ckpt=os.path.basename(a.ckpt), sha256=h, top_level_keys=list(c), iter=int(c['iter']), obs_dim=obs, action_dim=act,
    actor_keys=len(am), critic_keys=len(c['critic_state_dict']), optimizer_present=True, adam_step=adam_step, lr=os_['param_groups'][0]['lr'],
    obs_normalizer_in_checkpoint=any('norm' in k.lower() for k in am), infos=c.get('infos'))))
