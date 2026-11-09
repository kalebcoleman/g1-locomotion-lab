#!/usr/bin/env python3
"""Audit a diag_trace.py npz: what the robots counted as successes are really doing. Usage: audit_trace.py <trace.npz> [out.json]"""
import sys, json
import numpy as np
d = np.load(sys.argv[1], allow_pickle=False); dt = float(d["dt"]); T, N = d["h"].shape; cat = d["category"]; CAT = ["side_a", "prone", "side_b", "supine"]
def q(a, ps=(5, 50, 95)): return [round(float(x), 3) for x in np.percentile(a, ps)] if len(a) else None
res = dict(steps=T, robots=N)
for name in ["strict", "v2", "v21"]:
    ok = d[name].astype(bool); ids = np.where(ok)[0]; res[name] = dict(count=int(ok.sum()), by_category={CAT[c]: int(ok[cat == c].sum()) for c in range(4)})
    if not len(ids): continue
    supp = d["supp"][:, ids] > .5
    first = supp.argmax(0); last = T - 1 - supp[::-1].argmax(0); frac = supp.mean(0)
    # longest continuous supported run
    runs = []
    for k in range(len(ids)):
        best = cur = 0
        for v in supp[:, k]: cur = cur + 1 if v else 0; best = max(best, cur)
        runs.append(best)
    runs = np.array(runs); hold = supp
    m = lambda key: np.array([d[key][:, i][supp[:, k]].mean() for k, i in enumerate(ids)])
    res[name].update(first_supported_s=q(first * dt), longest_run_s=q(runs * dt), supported_last_step=int(supp[-1].sum()), supported_last_1s=int((supp[-50:].mean(0) > .9).sum()), supported_frac_of_episode=q(frac),
        hold_pelvis_h=q(m("h")), hold_up=q(m("up")), hold_speed=q(m("spd")), hold_stance_m=q(m("stance")), hold_foot_L_N=q(m("f_foot_l")), hold_foot_R_N=q(m("f_foot_r")), hold_hand_N=q(m("f_hand")), hold_arm_N=q(m("f_arm")), hold_knee_N=q(m("f_knee")), hold_trunk_N=q(m("f_trunk")), hold_hip_ankle_N=q(m("f_hip_ankle")),
        end_pelvis_h=q(d["h"][-1, ids]), end_up=q(d["up"][-1, ids]), drift_m=q(np.hypot(d["x"][-1, ids] - d["x"][0, ids], d["y"][-1, ids] - d["y"][0, ids])),
        tau_ratio_max=q(d["tau_ratio"][:, ids].max(0)), tau_over_real_frac=q(d["tau_over"][:, ids].mean(0)), tau_ok_97pct=int((d["tau_over"][:, ids].mean(0) <= .03).sum()),
        soft=q(d["soft_frac"][ids]), hard=q(d["hard_frac"][ids]), hard_ex=q(d["hard_excess"][ids]), flight=q(d["flight_frac"][ids]), vel=q(d["vel_frac"][ids]), peak_vz=q(d["peak_vz"][ids]))
    # validity over the rising window only: reset until the first supported step
    rw_soft, rw_hard, rw_tau, rw_len = [], [], [], []
    for k, i in enumerate(ids):
        e = max(first[k], 1); rw_soft.append((d["soft_ex"][:e, i] > 0).mean()); rw_hard.append((d["hard_ex"][:e, i] > 0).mean()); rw_tau.append(d["tau_over"][:e, i].mean()); rw_len.append(e * dt)
    res[name].update(rise_window_s=q(np.array(rw_len)), rise_soft_frac=q(np.array(rw_soft)), rise_hard_frac=q(np.array(rw_hard)), rise_tau_over_frac=q(np.array(rw_tau)), rise_soft_le10=int((np.array(rw_soft) <= .10).sum()), rise_hard_le3=int((np.array(rw_hard) <= .03).sum()))
jn = [str(x) for x in d["joint_names"]]; tm = d["tau_max"]; ov = d["tau_over_frac_j"]; lim = d["lim_real"]; ids = np.where(d["v21"].astype(bool))[0] if d["v21"].any() else np.where(d["strict"].astype(bool))[0]
res["joint_torque_on_audited"] = {n: dict(limit=float(lim[j]), max_Nm_p95=round(float(np.percentile(tm[ids, j], 95)), 1), over_frac_p95=round(float(np.percentile(ov[ids, j], 95)), 3)) for j, n in enumerate(jn) if np.isfinite(lim[j])}
res["joint_hand_tau_max_Nm"] = round(float(tm[:, ~np.isfinite(lim)].max()), 1)
res["geoms"] = {str(b): [int(t), [round(float(x), 3) for x in s]] for b, t, s in zip(d["geom_body"], d["geom_type"], d["geom_scale"])}
res["v21_ids"] = np.where(d["v21"].astype(bool))[0].tolist()
s = json.dumps(res, indent=0); 
if len(sys.argv) > 2: open(sys.argv[2], "w").write(s)
print(s)
