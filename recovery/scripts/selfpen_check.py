#!/usr/bin/env python3
"""Amendment 3b: offline self-penetration check on a diag_trace.py npz (the simulator runs with self-collision off, so limbs can pass through each other).
Rebuilds the robot's 38 collision shapes in MuJoCo as free-floating convex geoms, poses them from the saved body poses (10 Hz), and measures signed distance with mj_geomDistance.
Body pairs within two joints of each other are skipped. Convex hulls are larger than the real links, so depth is an upper bound.
Usage (lab python): selfpen_check.py <trace.npz> <out.json> [--which v21|strict|all] [--depth 0.02]"""
import sys, json, argparse
import numpy as np
import mujoco

ap = argparse.ArgumentParser(); ap.add_argument("trace"); ap.add_argument("out"); ap.add_argument("--which", default="v21"); ap.add_argument("--depth", type=float, default=0.02)
a = ap.parse_args(); d = np.load(a.trace, allow_pickle=False)
names = [str(x) for x in d["body_names"]]; gb = [str(x) for x in d["geom_body"]]; gt = d["geom_type"]; gs = d["geom_scale"]; gx = d["geom_xform"]; vo = d["geom_vert_off"]; fo = d["geom_face_off"]; V = d["geom_verts"]; F = d["geom_faces"]
spec = mujoco.MjSpec(); keep = []
for i in range(len(gb)):
    b = spec.worldbody.add_body(name=f"s{i}", mocap=True); t = int(gt[i]); kw = {}
    if vo[i + 1] > vo[i]:
        v = V[vo[i]:vo[i + 1]] * gs[i]; f = F[fo[i]:fo[i + 1]]
        m = spec.add_mesh(name=f"m{i}"); m.uservert = v.astype(np.float64).ravel().tolist()
        if len(f) >= 3: m.userface = f.astype(int).ravel().tolist()
        b.add_geom(name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"m{i}")
    elif t == 7:
        b.add_geom(name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[float(x) for x in gs[i]])
    elif t == 6:
        b.add_geom(name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[float(gs[i][0]), float(gs[i][1]), 0.0])
    elif t == 4:
        b.add_geom(name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[float(gs[i][0]), float(gs[i][1]), 0.0])
    elif t == 3:
        b.add_geom(name=f"g{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(gs[i][0]), 0.0, 0.0])
    else:
        raise SystemExit(f"shape {i} on {gb[i]} has type {t} and no vertices")
    keep.append(i)
model = spec.compile(); data = mujoco.MjData(model)

# kinematic graph from the G1 link names
E = [("pelvis", "pelvis_contour_link"), ("pelvis", "torso_link"), ("torso_link", "head_link"), ("torso_link", "imu_link"), ("torso_link", "logo_link")]
for s in ("left", "right"):
    leg = ["pelvis"] + [f"{s}_{n}_link" for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    arm = ["torso_link"] + [f"{s}_{n}_link" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "elbow_roll", "palm")]
    E += list(zip(leg, leg[1:])) + list(zip(arm, arm[1:]))
    E += [(f"{s}_palm_link", f"{s}_zero_link"), (f"{s}_zero_link", f"{s}_one_link"), (f"{s}_one_link", f"{s}_two_link"), (f"{s}_palm_link", f"{s}_three_link"), (f"{s}_three_link", f"{s}_four_link"), (f"{s}_palm_link", f"{s}_five_link"), (f"{s}_five_link", f"{s}_six_link")]
nbr = {n: set() for n in names}
for x, y in E:
    if x in nbr and y in nbr: nbr[x].add(y); nbr[y].add(x)
def near(x, y):
    if x == y or y in nbr[x]: return True
    return any(y in nbr[z] for z in nbr[x])
hand = lambda n: any(w in n.split("_") for w in ("palm", "zero", "one", "two", "three", "four", "five", "six"))
pairs = [(i, j) for i in range(len(gb)) for j in range(i + 1, len(gb)) if not near(gb[i], gb[j]) and not (hand(gb[i]) and hand(gb[j]) and gb[i].split("_")[0] == gb[j].split("_")[0])]

def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz])
def qrot(q, v):
    u = q[:3]; return v + 2 * np.cross(u, np.cross(u, v) + q[3] * v)

ids = np.where(d[a.which].astype(bool))[0] if a.which != "all" else np.arange(d["body_p"].shape[1])
bp = d["body_p"]; bq = d["body_q"]; idx = {n: k for k, n in enumerate(names)}; res = {}; pair_hits = {}
for env in ids:
    worst = 0.0; frames = 0
    for t in range(bp.shape[0]):
        for i in range(len(gb)):
            b = idx[gb[i]]; p = bp[t, env, b]; q = bq[t, env, b]; lp = gx[i][:3]; lq = gx[i][3:]
            wp = p + qrot(q, lp); wq = qmul(q, lq)
            data.mocap_pos[i] = wp; data.mocap_quat[i] = [wq[3], wq[0], wq[1], wq[2]]
        mujoco.mj_kinematics(model, data); hit = False
        for i, j in pairs:
            dist = mujoco.mj_geomDistance(model, data, i, j, 0.05, None)
            if dist < 0:
                worst = min(worst, dist)
                if dist < -a.depth:
                    hit = True; k = f"{gb[i]}|{gb[j]}"; pair_hits[k] = pair_hits.get(k, 0) + 1
        frames += hit
    res[int(env)] = dict(max_depth_m=round(-float(worst), 4), frames_over_depth=int(frames), frac_frames=round(frames / bp.shape[0], 3))
frac = np.array([r["frac_frames"] for r in res.values()]) if res else np.zeros(0)
out = dict(which=a.which, robots=len(ids), depth_threshold_m=a.depth, pairs_checked=len(pairs), sampled_hz=10, robots_clean=int((frac == 0).sum()), robots_le_3pct_frames=int((frac <= 0.03).sum()), max_depth_p50=float(np.median([r["max_depth_m"] for r in res.values()])) if res else None,
           top_pairs=dict(sorted(pair_hits.items(), key=lambda kv: -kv[1])[:12]), per_env=res, note="Convex-hull upper bound on depth, sampled at 10 Hz; fast crossings can be missed.")
json.dump(out, open(a.out, "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "per_env"}))
