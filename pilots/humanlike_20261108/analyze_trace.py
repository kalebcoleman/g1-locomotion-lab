"""Derive spread/hip telemetry from a diag_trace.py npz (runtime artifact). usage: analyze_trace.py trace.npz out.json
Foot spread = stance (m, ankle-roll link xy distance) per step (exact). Thigh abduction = frontal-plane angle of pelvis->knee vector in the pelvis frame, from body poses sampled every 5th step (DERIVED, not the hip-roll joint angle). Final hip roll/yaw = q_end (exact)."""
import sys, json, numpy as np
d = np.load(sys.argv[1]); bn = [str(x) for x in d['body_names']]; jn = [str(x) for x in d['joint_names']]
st = d['stance']; N = st.shape[1]; dt = float(d['dt'])
def Rm(q):  # (...,4) wxyz -> rotation matrices
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3,3))
bp, bq = d['body_p'], d['body_q']; pel = bn.index('pelvis')
R = Rm(bq[:, :, pel]); Rt = np.swapaxes(R, -1, -2)
def abd(side):
    k = bp[:, :, bn.index(f'{side}_knee_link')] - bp[:, :, pel]; v = np.einsum('tnij,tnj->tni', Rt, k)   # pelvis frame: x fwd, y left, z up
    lat = v[..., 1] * (1 if side == 'left' else -1)    # positive = leg moves outward
    return np.degrees(np.arctan2(lat, -v[..., 2]))
aL, aR = abd('left'), abd('right'); sp = aL + aR      # total thigh splay angle between the legs in the frontal plane (deg)
sup = d['strict'].astype(bool); v3 = d['v21'].astype(bool)
q_end = d['q_end']; hr = [jn.index(f'{s}_hip_roll_joint') for s in ('left', 'right')]; hy = [jn.index(f'{s}_hip_yaw_joint') for s in ('left', 'right')]
last = int(round(2.0 / dt))
def pack(x, m=None):
    x = x if m is None else x[m]; return dict(n=int(len(x)), mean=float(np.mean(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)), max=float(np.max(x))) if len(x) else None
h = d['h']; up = d['up']; rise = h > 0.40            # gate: pelvis above 0.40 m (the stance_excess gate); lying start excluded
rise5 = rise[::5][:sp.shape[0]]; upr5 = (rise & (up > 0.9))[::5][:sp.shape[0]]
def gmax(x, m): return np.where(m, x, np.nan)
def fin(x, m): return np.array([np.nanmean(c[-5:]) if m[-5:, i].any() else np.nan for i, c in enumerate(x.T)])
sp_up = gmax(sp, upr5); spg = np.nanmax(np.where(np.isnan(sp_up), -1e9, sp_up), 0); spg[spg < -1e8] = np.nan
stg = np.where(rise, st, -1.0).max(0); stg[stg < 0] = np.nan; stf = st[-int(1/dt):].mean(0)
out_g = dict(gate='pelvis>0.40 m (foot spread) ; pelvis>0.40 & up>0.9 (thigh splay)', foot_spread_peak_after_rise_m=dict(mean=float(np.nanmean(stg)), median=float(np.nanmedian(stg)), p90=float(np.nanpercentile(stg, 90)), max=float(np.nanmax(stg)), n_never_rose=int(np.isnan(stg).sum())),
    thigh_splay_peak_upright_deg=dict(mean=float(np.nanmean(spg)), median=float(np.nanmedian(spg)), p90=float(np.nanpercentile(spg, 90)), max=float(np.nanmax(spg)), n=int((~np.isnan(spg)).sum())),
    time_foot_spread_above_0p45m_while_risen_frac=float(((st > 0.45) & rise).sum() / max(rise.sum(), 1)))
cat = d['category']; names = ['side_a', 'prone', 'side_b', 'supine']
out = dict(n=N, dt=dt, definition='peak = max over the 10 s episode; final = mean over last 1 s; foot spread in m; thigh splay in deg (derived)',
    foot_spread_peak_m=pack(st.max(0)), foot_spread_final_m=pack(st[-int(1/dt):].mean(0)), foot_spread_peak_m_strict_success=pack(st.max(0), sup),
    thigh_splay_peak_deg=pack(sp.max(0)), thigh_splay_final_deg=pack(sp[-10:].mean(0)),
    hip_roll_final_abs_rad_mean_LR=float(np.abs(q_end[:, hr]).mean()), hip_yaw_final_abs_rad_mean_LR=float(np.abs(q_end[:, hy]).mean()),
    hard_frac_by_joint_top=sorted(((jn[j], float(d['hard_frac_j'][:, j].mean())) for j in range(len(jn))), key=lambda x: -x[1])[:5],
    gated=out_g, hard_excess_max_rad=float(d['hard_excess'].max()), hard_frac_mean=float(d['hard_frac'].mean()), soft_frac_mean=float(d['soft_frac'].mean()), tau_over_mean=float(d['tau_over'].mean()), flight_frac_mean=float(d['flight_frac'].mean()),
    by_category={n: dict(foot_spread_peak_m_mean=float(st.max(0)[cat == i].mean()), foot_spread_final_m_mean=float(st[-int(1/dt):].mean(0)[cat == i].mean()), thigh_splay_peak_deg_mean=float(sp.max(0)[cat == i].mean())) for i, n in enumerate(names)},
    smoothness=dict(note='joint acceleration/jerk/action-rate NOT saved by trace; missing metric'))
json.dump(out, open(sys.argv[2], 'w'), indent=1); print(json.dumps(out['gated'], indent=1)); print(json.dumps({k: out[k] for k in ['foot_spread_peak_m', 'foot_spread_final_m', 'thigh_splay_peak_deg', 'thigh_splay_final_deg', 'hip_roll_final_abs_rad_mean_LR', 'hip_yaw_final_abs_rad_mean_LR', 'hard_frac_by_joint_top', 'hard_excess_max_rad']}, indent=1))
print(json.dumps(out['by_category'], indent=1))
