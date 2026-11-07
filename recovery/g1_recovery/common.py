"""Shared pieces of the G1 get-up (fall recovery) task: fallen-pose sampling, success metrics, held-out poses.

Everything lane-independent lives here so that Lane A and Lane B differ ONLY in rewards / curriculum / extra events.
Conventions: quaternions are (x, y, z, w) like Isaac Lab's write_root_pose_to_sim_index.
"""
from __future__ import annotations
import math, os
import numpy as np
import torch

STAND_H = 0.74          # default pelvis height of the G1 in Isaac Lab (m)
SUCCESS_H = 0.60        # pelvis height for "standing" (m)
SUCCESS_TILT_COS = 0.90  # torso up-vector . world-up >= 0.90 (tilt <= ~26 deg)
HOLD_S = 2.0            # standing must be held this long to count as recovered
EPISODE_S = 10.0
SUPPORT_FORCE_N = 10.0      # each foot must carry at least this contact force
SUPPORT_FOOT_Z = 0.15       # feet must be within this height of the floor (m)
SUPPORT_PELVIS_SPEED = 0.6  # pelvis speed limit for a 'stable' stand (m/s)
# Curriculum levels (v3): 0 = standing, 1 = sit/crouch from the state bank, 2 = mid-rise states from the bank, 3/4 = lying on the ground.
# Levels 0, 3, 4 use the tilt sampler (alpha = tilt from upright, deg); levels 1, 2 draw from the bank (BANK_PATH).
LEVEL_ALPHA_MIN = [0.0, 0.0, 0.0, 70.0, 80.0]
LEVEL_ALPHA_MAX = [5.0, 5.0, 5.0, 90.0, 90.0]
LEVEL_JOINT_NOISE = [0.05, 0.0, 0.0, 0.3, 0.4]     # rad
BANK_PATH = os.environ.get("G1R_BANK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bank_v1.npz"))
NUM_LEVELS = 5
HELDOUT_SEED = 20261106
HELDOUT_N = 256
CATEGORIES = ["side_a", "prone", "side_b", "supine"]   # fall direction phi = 0, 90, 180, 270 deg


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
                        aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz], -1)


def tilt_yaw_quat(alpha, phi, yaw):
    """Rotation = yaw about world z applied after a tilt of `alpha` about the horizontal body axis (cos phi, sin phi, 0)."""
    s, c = torch.sin(alpha / 2), torch.cos(alpha / 2)
    qt = torch.stack([torch.cos(phi) * s, torch.sin(phi) * s, torch.zeros_like(s), c], -1)
    qy = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
    return quat_mul(qy, qt)


def sample_poses(n, alpha_min_deg, alpha_max_deg, joint_noise, default_q, q_lo, q_hi, noise_mask, gen, device, phi=None):
    """Return root z, root quat, joint positions for n fallen poses (all tensors on `device`).
    alpha_*: tensors (n,) in degrees; joint_noise: (n,) rad; noise_mask: (J,) 1 for joints that get noise."""
    u = lambda *shape: torch.rand(*shape, generator=gen, device="cpu").to(device)
    alpha = (alpha_min_deg + (alpha_max_deg - alpha_min_deg) * u(n)) * math.pi / 180.0
    if phi is None:
        phi = u(n) * 2 * math.pi
    yaw = u(n) * 2 * math.pi
    quat = tilt_yaw_quat(alpha, phi, yaw)
    z = 0.20 + 0.55 * torch.cos(alpha)                    # clear the floor, the robot settles in the first steps
    noise = (u(n, default_q.shape[-1]) * 2 - 1) * joint_noise[:, None] * noise_mask[None, :]
    q = torch.max(torch.min(default_q[None, :] + noise, q_hi[None, :]), q_lo[None, :])
    return z, quat, q


def make_heldout(default_q, q_lo, q_hi, noise_mask, device="cpu", path=None):
    """Deterministic held-out set: HELDOUT_N lying poses, 4 fall directions x 64, alpha in [80, 90] deg,
    joint noise 0.4 rad, a seed that the training sampler never uses. Saved as npz if `path` is given."""
    gen = torch.Generator().manual_seed(HELDOUT_SEED)
    n = HELDOUT_N
    cat = torch.arange(n) % 4
    phi = (cat.float() * 90.0 + (torch.rand(n, generator=gen) * 2 - 1) * 15.0) * math.pi / 180.0
    z, quat, q = sample_poses(n, torch.full((n,), 80.0), torch.full((n,), 90.0), torch.full((n,), 0.4), default_q.cpu(),
                              q_lo.cpu(), q_hi.cpu(), noise_mask.cpu(), gen, "cpu", phi=phi)
    d = dict(z=z.numpy(), quat=quat.numpy(), q=q.numpy(), category=cat.numpy(), seed=HELDOUT_SEED)
    if path:
        np.savez(path, **d)
    return d


def make_set(seed, default_q, q_lo, q_hi, noise_mask, noise_levels=(0.1, 0.4), n_per=32, path=None):
    """Evaluation set v2: 4 fall directions x len(noise_levels) limb-noise levels x n_per poses (default 4*2*32 = 256), alpha in [80,90] deg.
    Arrays: z, quat, q, category (direction 0..3), difficulty (index into noise_levels), noise (rad)."""
    gen = torch.Generator().manual_seed(seed); cats, diffs, zs, quats, qs = [], [], [], [], []
    for d in range(4):
        for li, nl in enumerate(noise_levels):
            n = n_per
            phi = (d * 90.0 + (torch.rand(n, generator=gen) * 2 - 1) * 15.0) * math.pi / 180.0
            z, quat, q = sample_poses(n, torch.full((n,), 80.0), torch.full((n,), 90.0), torch.full((n,), float(nl)), default_q.cpu(), q_lo.cpu(), q_hi.cpu(), noise_mask.cpu(), gen, "cpu", phi=phi)
            cats += [d] * n; diffs += [li] * n; zs.append(z); quats.append(quat); qs.append(q)
    perm = torch.randperm(len(cats), generator=gen)       # shuffle so any prefix of poses mixes all families
    arr = lambda v: torch.cat(v)[perm].numpy()
    d_ = dict(z=arr(zs), quat=arr(quats), q=arr(qs), category=torch.tensor(cats)[perm].numpy(), difficulty=torch.tensor(diffs)[perm].numpy(), noise=torch.tensor([noise_levels[i] for i in diffs])[perm].numpy(), seed=seed)
    if path: np.savez(path, **d_)
    return d_


class Tracker:
    """Per-env running recovery metrics, updated every env step by the `track_recovery` termination term."""
    FAIL = ["success", "never_rose", "rose_not_upright", "stood_then_fell", "late_hold_incomplete"]

    def __init__(self, n, device, step_dt):
        self.n, self.dev, self.dt = n, device, step_dt
        self.hold_steps_needed = int(round(HOLD_S / step_dt))
        z = lambda dt=torch.float32: torch.zeros(n, device=device, dtype=dt)
        self.reset_all = lambda: [setattr(self, k, v) for k, v in dict(
            step=z(torch.int32), run=z(torch.int32), run_start=z(torch.int32), best_run=z(torch.int32), total_stand=z(torch.int32),
            max_h=z(), succ=z(torch.bool), succ_start=torch.full((n,), -1, device=device, dtype=torch.int32),
            ever_stood=z(torch.bool), viol_steps=z(torch.int32), viol_joint_steps=z(torch.int32),
            stand_now=z(torch.bool), last_h=z(), last_up=z(), vel_viol_steps=z(torch.int32), peak_vz=z(), flight_steps=z(torch.int32),
            run_s=z(torch.int32), best_run_s=z(torch.int32), total_stand_s=z(torch.int32), run_start_s=z(torch.int32), succ_s=z(torch.bool),
            succ_start_s=torch.full((n,), -1, device=device, dtype=torch.int32), ever_stood_s=z(torch.bool), stand_now_s=z(torch.bool)).items()]
        self.reset_all()

    def reset(self, ids):
        for k in ["step", "run", "run_start", "best_run", "total_stand", "viol_steps", "viol_joint_steps", "run_s", "best_run_s", "total_stand_s", "run_start_s", "vel_viol_steps", "flight_steps"]:
            getattr(self, k)[ids] = 0
        for k in ["max_h", "last_h", "last_up", "peak_vz"]:
            getattr(self, k)[ids] = 0.0
        for k in ["succ", "ever_stood", "stand_now", "succ_s", "ever_stood_s", "stand_now_s"]:
            getattr(self, k)[ids] = False
        self.succ_start[ids] = -1; self.succ_start_s[ids] = -1

    def update(self, h, up, viol_count, supported=None, vel_viol=None, vz=None, flight=None):
        if flight is not None: self.flight_steps += flight.int()
        if vel_viol is not None: self.vel_viol_steps += vel_viol.int()
        if vz is not None: self.peak_vz = torch.maximum(self.peak_vz, vz)
        stand = (h >= SUCCESS_H) & (up >= SUCCESS_TILT_COS)
        if supported is not None:      # strict, feet-supported stand
            ss = stand & supported
            new_s = ss & ~self.stand_now_s
            self.run_start_s = torch.where(new_s, self.step, self.run_start_s)
            self.run_s = torch.where(ss, self.run_s + 1, torch.zeros_like(self.run_s)); self.best_run_s = torch.maximum(self.best_run_s, self.run_s)
            self.total_stand_s += ss.int()
            newly_s = (self.run_s >= self.hold_steps_needed) & ~self.succ_s
            self.succ_start_s = torch.where(newly_s, self.run_start_s, self.succ_start_s); self.succ_s |= newly_s
            self.ever_stood_s |= ss; self.stand_now_s = ss
        new_run = stand & ~self.stand_now
        self.run_start = torch.where(new_run, self.step, self.run_start)
        self.run = torch.where(stand, self.run + 1, torch.zeros_like(self.run))
        self.best_run = torch.maximum(self.best_run, self.run)
        self.total_stand += stand.int()
        newly = (self.run >= self.hold_steps_needed) & ~self.succ
        self.succ_start = torch.where(newly, self.run_start, self.succ_start)
        self.succ |= newly
        self.ever_stood |= stand
        self.max_h = torch.maximum(self.max_h, h)
        self.viol_steps += (viol_count > 0).int(); self.viol_joint_steps += viol_count.int()
        self.stand_now = stand; self.last_h = h; self.last_up = up
        self.step += 1

    def failure_cause(self):
        """Index into FAIL for each env (0 = success)."""
        c = torch.full((self.n,), 4, device=self.dev, dtype=torch.int64)       # late_hold_incomplete (default)
        c = torch.where(self.ever_stood & ~self.stand_now, torch.full_like(c, 3), c)   # stood_then_fell
        c = torch.where(~self.ever_stood & (self.max_h >= 0.35), torch.full_like(c, 2), c)  # rose_not_upright
        c = torch.where(~self.ever_stood & (self.max_h < 0.35), torch.full_like(c, 1), c)   # never_rose
        c = torch.where(self.succ, torch.zeros_like(c), c)
        return c


# ----------------------------------------------------------------------------- env-side functions (MDP terms)
HAND_KEYS = ("five", "three", "six", "four", "zero", "one", "two")
def is_hand_joint(name):
    return any(k in name for k in HAND_KEYS)


def get_tracker(env):
    if not hasattr(env, "_rec"):
        env._rec = Tracker(env.num_envs, env.device, env.step_dt)
    return env._rec


def get_bank(env):
    """State bank (root z, quat, joints) harvested from rollouts; levels 1/2 sample from it. None if the file is missing (levels fall back to tilt poses)."""
    if not hasattr(env, "_rec_bank"):
        env._rec_bank = None
        if os.path.exists(BANK_PATH):
            f = np.load(BANK_PATH); dev = env.device
            z = torch.as_tensor(f["z"], device=dev).float(); quat = torch.as_tensor(f["quat"], device=dev).float(); q = torch.as_tensor(f["q"], device=dev).float()
            up = 1.0 - 2.0 * (quat[:, 0] ** 2 + quat[:, 1] ** 2)            # world-up component of the body z axis
            sit = torch.nonzero((up >= 0.7) & (z >= 0.15) & (z <= 0.55)).squeeze(-1)
            mid = torch.nonzero((z >= 0.12) & (z <= 0.62)).squeeze(-1)
            env._rec_bank = dict(z=z, quat=quat, q=q, sit=sit, mid=mid)
    return env._rec_bank


def get_levels(env):
    if not hasattr(env, "_rec_level"):
        env._rec_level = torch.full((env.num_envs,), int(getattr(env.cfg, "start_level", None) if getattr(env.cfg, "start_level", None) is not None else os.environ.get("G1R_START_LEVEL", "0")), device=env.device, dtype=torch.long)
    return env._rec_level


def _robot_h_up(env):
    r = env.scene["robot"]
    h = r.data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    up = -r.data.projected_gravity_b.torch[:, 2]
    return h, up


CRITERIA_VERSION = os.environ.get("G1R_CRITERIA", "v2")     # v1: pelvis speed = xy only (all results before 2026-11-06 21:30); v2: 3D speed, position-limit validity (see EVAL_PROTOCOL.md)
STRICT = os.environ.get("G1R_STRICT", "0") == "1"   # training reward/curriculum use the feet-supported stand criterion


def supported_mask(env):
    """Both feet carrying contact force on the floor, feet near the floor, pelvis speed low (see SUPPORT_* constants)."""
    r = env.scene["robot"]
    if not hasattr(env, "_feet_idx"):
        sens = env.scene.sensors["contact_forces"]; names = list(sens.body_names)
        env._feet_idx = [i for i, n in enumerate(names) if "ankle_roll" in n]
        env._feet_body = [r.body_names.index(n) for n in r.body_names if "ankle_roll" in n]
    f = torch.linalg.norm(env.scene.sensors["contact_forces"].data.net_forces_w.torch[:, env._feet_idx], dim=-1)
    fz = r.data.body_pos_w.torch[:, env._feet_body, 2] - env.scene.env_origins[:, 2:3]
    v = r.data.root_lin_vel_w.torch
    spd = torch.linalg.norm(v if CRITERIA_VERSION == "v2" else v[:, :2], dim=-1)
    return (f > SUPPORT_FORCE_N).all(-1) & (fz < SUPPORT_FOOT_Z).all(-1) & (spd < SUPPORT_PELVIS_SPEED)


def track_recovery(env, env_ids=None):
    """Termination term (never terminates): updates the recovery metrics once per env step."""
    r = env.scene["robot"]
    h, up = _robot_h_up(env)
    q = r.data.joint_pos.torch; lim = r.data.soft_joint_pos_limits.torch
    if not hasattr(env, "_body_jmask"):   # finger joints are excluded: their default pose lies outside their soft limits
        env._body_jmask = torch.tensor([not is_hand_joint(n) for n in r.joint_names], device=env.device)
    viol = (((q < lim[..., 0]) | (q > lim[..., 1])) & env._body_jmask).sum(-1)
    supported = supported_mask(env)
    if not hasattr(env, "_vel_lim"):
        env._vel_lim = r.data.soft_joint_vel_limits.torch[0].clone()
    vv = ((r.data.joint_vel.torch.abs() > env._vel_lim[None]) & env._body_jmask).any(-1)
    fl = (torch.linalg.norm(env.scene.sensors["contact_forces"].data.net_forces_w.torch, dim=-1).sum(-1) < 1.0)    # no body touching anything = free flight
    get_tracker(env).update(h, up, viol, supported, vv, r.data.root_lin_vel_w.torch[:, 2], fl)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def standing_mask(env):
    h, up = _robot_h_up(env)
    m = (h >= SUCCESS_H) & (up >= SUCCESS_TILT_COS)
    return (m & supported_mask(env)) if STRICT else m


def reset_fallen(env, env_ids, fixed_poses: str | None = None):
    """Reset event: put the robot on the ground in a pose drawn for each env's curriculum level (or from the held-out file)."""
    r = env.scene["robot"]; dev = env.device
    ids = env_ids if torch.is_tensor(env_ids) else torch.arange(env.num_envs, device=dev)[env_ids]
    n = len(ids)
    defq = r.data.default_joint_pos.torch[0]; lim = r.data.soft_joint_pos_limits.torch[0]
    mask = torch.ones_like(defq)
    for i, nm in enumerate(r.joint_names):
        if is_hand_joint(nm):
            mask[i] = 0.0       # hand joints stay at their default
    fixed = getattr(env, "_rec_fixed", None)
    if fixed is None and fixed_poses is not None and os.path.exists(fixed_poses):
        f = np.load(fixed_poses)
        env._rec_fixed = fixed = {k: torch.as_tensor(f[k], device=dev) for k in ["z", "quat", "q", "category"]}
    if fixed is not None:
        sel = ids % fixed["z"].shape[0]
        z, quat, q = fixed["z"][sel].float(), fixed["quat"][sel].float(), fixed["q"][sel].float()
    else:
        if not hasattr(env, "_rec_gen"):
            env._rec_gen = torch.Generator().manual_seed(int(env.cfg.seed or 0) + 1000003)   # != HELDOUT_SEED stream
        lv = get_levels(env)[ids]
        amin = torch.tensor(LEVEL_ALPHA_MIN, device=dev)[lv]; amax = torch.tensor(LEVEL_ALPHA_MAX, device=dev)[lv]
        jn = torch.tensor(LEVEL_JOINT_NOISE, device=dev)[lv]
        z, quat, q = sample_poses(n, amin, amax, jn, defq, lim[:, 0], lim[:, 1], mask, env._rec_gen, dev)
        bank = get_bank(env)
        if bank is not None:
            for L, sel in ((1, bank["sit"]), (2, bank["mid"])):
                m = lv == L
                k = int(m.sum())
                if k and len(sel):
                    pick = sel[torch.randint(len(sel), (k,), generator=env._rec_gen).to(dev)]
                    yaw = torch.rand(k, generator=env._rec_gen).to(dev) * 2 * math.pi
                    qy = torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2), torch.cos(yaw / 2)], -1)
                    z[m] = bank["z"][pick] + 0.02; quat[m] = quat_mul(qy, bank["quat"][pick]); q[m] = bank["q"][pick]
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :2] = env.scene.env_origins[ids, :2]; pose[:, 2] = env.scene.env_origins[ids, 2] + z; pose[:, 3:] = quat
    r.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    r.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=ids)
    r.write_joint_position_to_sim_index(position=q, env_ids=ids)
    r.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=ids)
    get_tracker(env).reset(ids)
