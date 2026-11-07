"""Multi-critic PPO ported from HoST (https://github.com/InternRobotics/HoST, MIT, commit 70bb580: rsl_rl/rsl_rl/algorithms/ppo.py,
modules/actor_critic.py, storage/rollout_storage.py), driving an Isaac Lab 3.0.0 env built from g1_recovery.host_port.
One critic per reward group (task, regu, style, target); per-critic GAE; advantages normalised per critic and combined with HoST's group
weights (2.5, 0.1, 1, 1); clipped surrogate and value losses; entropy 0.01; HoST's smoothness regulariser on interpolated observations;
adaptive learning rate (desired KL 0.01, start 1e-3); 5 epochs x 4 minibatches; 50 steps per env per iteration.
  run.sh train_host.py --task G1-Host-Supine-v0 --num_envs 8192 --max_iterations 6000 --seed 3 --out_dir <abs dir> [--resume ckpt] physics=newton_mjwarp"""
import sys, os, json, time, argparse
import warp as wp
wp.config.enable_backward = False
import torch
import torch.nn as nn
from torch.distributions import Normal
from isaaclab.app import launch_simulation
from isaaclab_rl.entrypoints.backends import play_rsl_rl as P
from isaaclab_rl.entrypoints import common as C

ex = argparse.ArgumentParser(add_help=False)
ex.add_argument("--out_dir", required=True); ex.add_argument("--max_iterations", type=int, default=6000); ex.add_argument("--resume", default=None)
ex.add_argument("--steps", type=int, default=50); ex.add_argument("--save_every", type=int, default=100)
xa, rest = ex.parse_known_args(sys.argv[1:])
args = P._parse_args(rest); args.train_env_cfg = True
import g1_recovery  # noqa: F401
from g1_recovery import host_port as H
from isaaclab_tasks.utils import resolve_task_config

ACTOR_H, CRITIC_H = [512, 256, 128], [512, 256]
GAMMA, LAM, CLIP, EPOCHS, MINIB, ENT, VCOEF, GRAD = 0.99, 0.95, 0.2, 5, 4, 0.01, 1.0, 1.0
KL, LR0, STD0 = 0.01, 1e-3, 0.8
SM_UP, SM_LO, SM_V = 1.0, 0.1, 0.1
EPS = SM_LO / (SM_UP - SM_LO); PCOEF = SM_UP * EPS; VSCOEF = SM_V * PCOEF
GW = torch.tensor(H.GROUP_WEIGHTS)

env_cfg, _ = resolve_task_config(args.task, args.agent, play_mode=False)
os.makedirs(xa.out_dir, exist_ok=True)
with launch_simulation(env_cfg, args):
    C.apply_env_overrides(args, env_cfg)
    env_cfg.seed = args.seed if args.seed is not None else 3
    env = C.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=False)
    u = env.unwrapped; dev = u.device; N = u.num_envs; T = xa.steps; dt = u.step_dt
    torch.manual_seed(env_cfg.seed)
    rm = u.reward_manager; names = list(rm._term_names)
    gidx = torch.tensor([H.GROUPS.index(n.split("_")[0]) for n in names], device=dev)
    G = len(H.GROUPS); GWd = GW.to(dev)
    obs_d, _ = u.reset(); obs = obs_d["policy"]; OD = obs.shape[1]; AD = u.action_manager.total_action_dim
    actor = H.mlp(OD, ACTOR_H, AD, tanh=True).to(dev)
    critics = nn.ModuleList([H.mlp(OD, CRITIC_H, 1) for _ in range(G)]).to(dev)
    std = nn.Parameter(STD0 * torch.ones(AD, device=dev))
    params = list(actor.parameters()) + list(critics.parameters()) + [std]
    opt = torch.optim.Adam(params, lr=LR0); lr = LR0; it0 = 0
    if xa.resume:
        ck = torch.load(xa.resume, map_location=dev, weights_only=False)
        actor.load_state_dict(ck["actor"]); critics.load_state_dict(ck["critics"]); std.data.copy_(ck["std"]); it0 = ck["iteration"] + 1
    V = lambda o: torch.cat([c(o) for c in critics], -1)
    log = open(os.path.join(xa.out_dir, "train_log.jsonl"), "a")
    json.dump(dict(task=args.task, num_envs=N, steps=T, obs_dim=OD, act_dim=AD, groups=H.GROUPS, group_weights=H.GROUP_WEIGHTS, terms=names, seed=env_cfg.seed,
                   actor_hidden=ACTOR_H, critic_hidden=CRITIC_H, source="HoST port, see g1_recovery/host_port.py"), open(os.path.join(xa.out_dir, "config.json"), "w"), indent=1)
    S_obs = torch.zeros(T + 1, N, OD, device=dev); S_act = torch.zeros(T, N, AD, device=dev); S_lp = torch.zeros(T, N, device=dev)
    S_mu = torch.zeros(T, N, AD, device=dev); S_sig = torch.zeros(T, N, AD, device=dev); S_val = torch.zeros(T, N, G, device=dev)
    S_rew = torch.zeros(T, N, G, device=dev); S_done = torch.zeros(T, N, device=dev)
    ep_ret = torch.zeros(N, G, device=dev); fin = []
    for it in range(it0, it0 + xa.max_iterations):
        t0 = time.time()
        with torch.no_grad():
            for s in range(T):
                S_obs[s] = obs
                mu = actor(obs); dist = Normal(mu, std.expand_as(mu)); a = dist.sample()
                S_act[s] = a; S_lp[s] = dist.log_prob(a).sum(-1); S_mu[s] = mu; S_sig[s] = std.expand_as(mu); S_val[s] = V(obs)
                obs_d, _, term, trunc, extras = u.step(a)
                obs = obs_d["policy"]
                r = torch.zeros(N, G, device=dev).index_add_(1, gidx, rm._step_reward * dt)
                done = (term | trunc).float()
                r = r + GAMMA * S_val[s] * trunc.float()[:, None]            # bootstrap on time-outs (HoST)
                S_rew[s] = r; S_done[s] = done
                ep_ret += r
                if done.any():
                    fin.append(ep_ret[done.bool()].mean(0).cpu()); ep_ret[done.bool()] = 0
            S_obs[T] = obs
            last = V(obs)
            adv = torch.zeros(T, N, G, device=dev); g = torch.zeros(N, G, device=dev)
            for s in reversed(range(T)):
                nv = last if s == T - 1 else S_val[s + 1]
                nt = (1.0 - S_done[s])[:, None]
                delta = S_rew[s] + nt * GAMMA * nv - S_val[s]
                g = delta + nt * GAMMA * LAM * g; adv[s] = g
            ret = adv + S_val
            advn = (adv - adv.mean((0, 1))) / (adv.std((0, 1)) + 1e-8)
            madv = (advn * GWd).sum(-1)
        # update (HoST uses T-1 transitions so each sample has a next observation)
        B = N * (T - 1); mb = B // MINIB
        f = lambda x: x[: T - 1].reshape(B, *x.shape[2:])
        o_b, on_b = S_obs[: T - 1].reshape(B, OD), S_obs[1:T].reshape(B, OD)
        cont = (1.0 - S_done[: T - 1]).reshape(B, 1)
        a_b, lp_b, mu_b0, sg_b0, v_b0, ret_b, adv_b = f(S_act), f(S_lp), f(S_mu), f(S_sig), f(S_val), f(ret), f(madv)
        st = dict(vloss=0.0, sloss=0.0, smooth=0.0, kl=0.0)
        for ep in range(EPOCHS):
            perm = torch.randperm(MINIB * mb, device=dev)
            for k in range(MINIB):
                idx = perm[k * mb:(k + 1) * mb]
                ob = o_b[idx]; mu = actor(ob); dist = Normal(mu, std.expand_as(mu))
                lp = dist.log_prob(a_b[idx]).sum(-1); ent = dist.entropy().sum(-1)
                vb = V(ob)
                with torch.no_grad():
                    sig = std.expand_as(mu)
                    kl = torch.sum(torch.log(sig / sg_b0[idx] + 1e-5) + (sg_b0[idx] ** 2 + (mu_b0[idx] - mu) ** 2) / (2.0 * sig ** 2) - 0.5, -1).mean()
                    if kl > KL * 2.0: lr = max(1e-5, lr / 1.5)
                    elif 0.0 < kl < KL / 2.0: lr = min(1e-2, lr * 1.5)
                    for pg in opt.param_groups: pg["lr"] = lr
                ratio = torch.exp(lp - lp_b[idx]); A = adv_b[idx]
                sloss = torch.max(-A * ratio, -A * torch.clamp(ratio, 1 - CLIP, 1 + CLIP)).mean()
                vcl = v_b0[idx] + (vb - v_b0[idx]).clamp(-CLIP, CLIP)
                vloss = torch.max((vb - ret_b[idx]) ** 2, (vcl - ret_b[idx]) ** 2).mean()
                mixw = cont[idx] * (torch.rand_like(cont[idx]) - 0.5) * 2.0
                mob = ob + mixw * (on_b[idx] - ob)
                smooth = PCOEF * torch.square(torch.norm(mu - actor(mob), dim=-1)).mean() + VSCOEF * torch.square(torch.norm(vb - V(mob), dim=-1)).mean()
                loss = sloss + VCOEF * vloss - ENT * ent.mean() + smooth
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, GRAD); opt.step()
                st["vloss"] += vloss.item(); st["sloss"] += sloss.item(); st["smooth"] += smooth.item(); st["kl"] += float(kl)
        n_up = EPOCHS * MINIB
        hst = u._host
        epi = {k.replace("Episode_Reward/", ""): float(v) for k, v in (extras.get("log", {}) or {}).items() if isinstance(v, (int, float)) or (torch.is_tensor(v) and v.numel() == 1)}
        rec = dict(it=it, sps=round(N * T / (time.time() - t0)), lr=lr, std=float(std.mean()), force=float(hst["force"].mean()), rescale=float(hst["rescale"].mean()),
                   group_return=(torch.stack(fin[-200:]).mean(0).tolist() if fin else None), **{k: round(v / n_up, 5) for k, v in st.items()},
                   head_h=float(hst["last_head"].mean()), episode=epi)
        fin = fin[-400:]
        log.write(json.dumps(rec) + "\n"); log.flush()
        if it % 10 == 0:
            print(f"HOST it {it} sps {rec['sps']} std {rec['std']:.3f} lr {lr:.2e} force {rec['force']:.1f} rescale {rec['rescale']:.3f} head {rec['head_h']:.3f} "
                  f"G {[round(x, 2) for x in rec['group_return']] if rec['group_return'] else None} v {rec['vloss']:.3f} kl {rec['kl']:.4f}", flush=True)
        if it % xa.save_every == 0 or it == it0 + xa.max_iterations - 1:
            torch.save(dict(actor=actor.state_dict(), critics=critics.state_dict(), std=std.data, optimizer=opt.state_dict(), iteration=it, obs_dim=OD, act_dim=AD,
                            actor_hidden=ACTOR_H, critic_hidden=CRITIC_H, task=args.task, groups=H.GROUPS), os.path.join(xa.out_dir, f"model_{it}.pt"))
    C.close_env(env)
