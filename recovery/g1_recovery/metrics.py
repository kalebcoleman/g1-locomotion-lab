"""Pure (simulator-free) scoring functions for recovery and recovery->walking chains, so they can be unit-tested against CRITERIA.md.
All tensors are batched over environments. Criteria version: v2 (see EVAL_PROTOCOL.md). v1 results keep their original definitions."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Params:
    stand_h: float = 0.60            # pelvis height (m)
    stand_up: float = 0.90           # torso up-vector . world-up (tilt <= ~26 deg)
    foot_force: float = 10.0         # N per foot
    foot_z: float = 0.15             # m, feet above floor
    speed_max: float = 0.6           # m/s, 3D pelvis (root COM) speed for a supported stand (v1 used xy only)
    hold_s: float = 2.0
    vel_viol_frac: float = 0.02      # joint-velocity-limit-violating steps
    peak_vz: float = 2.0             # m/s peak pelvis upward speed
    flight_frac: float = 0.02        # free-flight steps
    pos_viol_frac: float = 0.10      # joint-POSITION-limit-violating steps (v2 addition)
    hard_viol_frac: float = 0.03     # v2.1: steps with any body joint past its HARD limit (walking reference 1.5%)
    hard_excess: float = 0.10        # v2.1: max excursion past a hard limit, rad (walking reference p99 0.083, max 0.105)
    torque_over_frac: float = 0.03   # Amendment 3a: steps with any body joint above Unitree's published torque limit
    persist_frac: float = 0.90       # Amendment 3c: supported on at least this share of the final second
    stance_max: float = 0.60         # Amendment 3d: mean distance between the feet over the 2 s hold, m
    walk_window_s: float = 10.0
    walk_dist: float = 3.0           # m along the commanded heading within the window
    walk_speed: float = 0.5          # m/s mean along heading
    fall_h: float = 0.45
    fall_up: float = 0.5


P = Params()


def supported_stand(h, up, feet_force, feet_z, speed3d, p: Params = P):
    """(N,) bool. feet_force/feet_z: (N,2). BOTH feet must carry force and be near the floor."""
    return (h >= p.stand_h) & (up >= p.stand_up) & (feet_force > p.foot_force).all(-1) & (feet_z < p.foot_z).all(-1) & (speed3d < p.speed_max)


def plausible_recovery(supported_success, vel_viol_steps, flight_steps, pos_viol_steps, peak_vz, steps, assist_active=None, p: Params = P):
    """Stage-1 plausible unassisted recovery (v2): supported 2 s hold AND velocity/flight/launch/position-limit validity AND no assist."""
    ok = supported_success & (vel_viol_steps.float() / steps <= p.vel_viol_frac) & (flight_steps.float() / steps <= p.flight_frac) \
        & (peak_vz <= p.peak_vz) & (pos_viol_steps.float() / steps <= p.pos_viol_frac)
    if assist_active is not None:
        ok = ok & ~assist_active
    return ok


# Unitree's published actuator limits (unitree_rl_lab, BSD-3, G1 29-DoF cfg), Nm. This USD's elbow_pitch / elbow_roll / torso_joint map to elbow / wrist_roll / waist_yaw. Finger joints have no published entry and are excluded.
REAL_TORQUE_LIMITS = (("hip_pitch", 88.0), ("hip_yaw", 88.0), ("hip_roll", 139.0), ("knee", 139.0), ("ankle", 50.0), ("torso_joint", 88.0), ("shoulder", 25.0), ("elbow", 25.0))


def real_torque_limit(joint_name):
    return next((v for k, v in REAL_TORQUE_LIMITS if k in joint_name), float("inf"))


def torque_ok(over_steps, steps, p: Params = P):
    """Amendment 3a: every body joint within the real limit on at least 97% of steps."""
    return over_steps.float() / steps <= p.torque_over_frac


def persists(supported_tail_frac, p: Params = P):
    """Amendment 3c: still in the supported stand during the final second of the episode."""
    return supported_tail_frac >= p.persist_frac


def stance_ok(stance_hold_mean, p: Params = P):
    """Amendment 3d: the counted hold is not a wide straddle."""
    return stance_hold_mean <= p.stance_max


def heading_progress(p0_xy, p1_xy, heading_xy):
    """Displacement projected on the (unit) commanded heading, and the lateral part. (N,2),(N,2),(N,2) -> (N,),(N,)"""
    d = p1_xy - p0_xy
    hn = heading_xy / heading_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    along = (d * hn).sum(-1)
    lateral = d[:, 0] * -hn[:, 1] + d[:, 1] * hn[:, 0]
    return along, lateral


def fell(h, up, p: Params = P):
    return (h < p.fall_h) | (up < p.fall_up)


def chain_success(handed_over, plausible_at_handover, window_complete, fell_in_window, along_disp, window_s, p: Params = P):
    """Stage-2 chain success (v2): plausible recovery -> hand-over, then within the walking window: window completed, no fall,
    >= walk_dist along the commanded heading, mean speed along heading >= walk_speed. Returns (success (N,), reason (N,) int)."""
    mean_speed = along_disp / window_s
    reason = torch.zeros_like(along_disp, dtype=torch.long)       # 0 = success
    reason = torch.where(~handed_over, torch.full_like(reason, 1), reason)                                   # never handed over
    reason = torch.where(handed_over & ~plausible_at_handover & (reason == 0), torch.full_like(reason, 2), reason)  # implausible recovery
    reason = torch.where(handed_over & ~window_complete & (reason == 0), torch.full_like(reason, 3), reason)         # timeout (window incomplete)
    reason = torch.where(fell_in_window & (reason == 0), torch.full_like(reason, 4), reason)                          # fell after hand-over
    reason = torch.where((along_disp < p.walk_dist) & (reason == 0), torch.full_like(reason, 5), reason)             # too little progress along heading
    reason = torch.where((mean_speed < p.walk_speed) & (reason == 0), torch.full_like(reason, 6), reason)            # too slow
    return reason == 0, reason


def hard_limits_ok(hard_viol_steps, hard_excess_max, steps, p: Params = P):
    """v2.1 addition: excursions past HARD joint limits stay near what the walking gait itself produces in this (compliant-limit) solver."""
    return (hard_viol_steps.float() / steps <= p.hard_viol_frac) & (hard_excess_max <= p.hard_excess)


POSTURES = ["trunk_off_floor", "seated_upright", "raised_not_upright_fold", "crouch_on_feet", "high_on_feet", "supported_stand_any", "supported_1s", "supported_2s"]


def posture_flags(h, up, feet_force, feet_z, trunk_force, speed3d, run_s, step_dt, p: Params = P):
    """Per-step boolean flags (N, len(POSTURES)) for the intermediate postures on the way to a stand; OR them over the episode to count how many robots reach each one.
    feet_ok = both feet carry force and are near the floor. fold = pelvis >= 0.35 m with the torso not upright (up < 0.5), the R6 optimum."""
    feet_ok = (feet_force > p.foot_force).all(-1) & (feet_z < p.foot_z).all(-1)
    return torch.stack([trunk_force < 1.0, (h >= 0.12) & (up >= 0.7), (h >= 0.35) & (up < 0.5), (h >= 0.35) & (up >= 0.7) & feet_ok, (h >= 0.50) & (up >= 0.7) & feet_ok,
                        supported_stand(h, up, feet_force, feet_z, speed3d, p), run_s * step_dt >= 1.0, run_s * step_dt >= p.hold_s], -1)


CHAIN_REASONS = ["success", "never_handed_over", "implausible_recovery", "timeout_window_incomplete", "fell_after_handover", "insufficient_progress_along_heading", "too_slow"]


# ---- Movement-quality metrics (M1). Pure functions over per-step traces; none of them changes a success criterion. ----
def movement_metrics(spread, h, hip_q, jvel, act, dt, rise_end=None, gate_low=0.12, gate=0.40, wide=0.45, final_s=1.0):
    """Per-robot movement descriptors from per-step traces. Time-major inputs, T steps, N robots.
    spread (T,N): floor-plane distance between the two ankle-roll links, m.  h (T,N): pelvis height above floor, m.
    hip_q (T,N,4): joint positions, order [hip_roll_left, hip_roll_right, hip_yaw_left, hip_yaw_right], rad (default pose is 0).
    jvel (T,N,J): body-joint velocities, rad/s (hand joints excluded by the caller).  act (T,N,A): policy actions applied at each step.
    rise_end (N,) long: first step NOT in the rising window (start of the counted hold; T if there was none). Default: T.
    Returns a dict of (N,) tensors:
      spread_peak_m          max spread over steps with h > gate_low (excludes the lying start); NaN if the pelvis never exceeds gate_low
      spread_peak_gated_m    same with h > gate (the stance penalty's gate); NaN if never
      spread_peak_time_s     time of spread_peak_m
      spread_final_m         mean spread over the last final_s seconds
      spread_wide_frac       share of steps with h > gate that have spread > wide
      hip_roll_peak_abs, hip_yaw_peak_abs      max over steps and both sides of |q|, rad
      hip_roll_final_abs, hip_yaw_final_abs    mean over the last final_s seconds and both sides of |q|, rad
      action_rate_mean       mean over the rising window and action dims of |a_t - a_(t-1)|  (policy-output units)
      joint_acc_rms, joint_acc_p99    RMS and 99th percentile of |d jvel / dt| over the rising window and joints, rad/s^2
      joint_jerk_p99         99th percentile of |d^2 jvel / dt^2| over the same, rad/s^3
    Joint acceleration is a finite difference of consecutive control-step velocities (dt = control step), so it under-reads accelerations shorter than dt."""
    T, N = spread.shape
    nan = torch.tensor(float("nan"), device=spread.device)
    re = torch.full((N,), T, device=spread.device, dtype=torch.long) if rise_end is None else rise_end.clamp(2, T)
    idx = torch.arange(T, device=spread.device)[:, None]
    g_low, g = h > gate_low, h > gate
    sp = torch.where(g_low, spread, torch.full_like(spread, float("-inf"))); pk, pk_i = sp.max(0)
    spg = torch.where(g, spread, torch.full_like(spread, float("-inf"))).max(0).values
    fn = max(1, int(round(final_s / dt)))
    q = hip_q.abs()
    win = idx < re[None]                                                    # (T,N) rising window
    acc = (jvel[1:] - jvel[:-1]).abs() / dt; jerk = (acc[1:] - acc[:-1]).abs() / dt     # acc[k] belongs to step k+1
    am = (idx[1:] < re[None])[..., None].expand_as(acc).float(); acc_rms = ((acc ** 2 * am).sum((0, 2)) / am.sum((0, 2)).clamp(min=1)).sqrt()
    da = (act[1:] - act[:-1]).abs().mean(-1); dm = (idx[1:] < re[None]).float()
    def wq2(x, mask_off, qq):                                               # x (T-k,N,J) with step offset mask_off
        m = (idx[mask_off: x.shape[0] + mask_off] < re[None])[..., None].expand_as(x)
        flat = torch.where(m, x, torch.full_like(x, float("nan"))).permute(1, 0, 2).reshape(N, -1)
        return torch.nanquantile(flat, qq, dim=1)
    return dict(spread_peak_m=torch.where(torch.isfinite(pk), pk, nan), spread_peak_gated_m=torch.where(torch.isfinite(spg), spg, nan),
                spread_peak_time_s=torch.where(torch.isfinite(pk), pk_i.float() * dt, nan), spread_final_m=spread[-fn:].mean(0),
                spread_wide_frac=((spread > wide) & g).float().sum(0) / g.float().sum(0).clamp(min=1),
                hip_roll_peak_abs=q[..., :2].amax((0, 2)), hip_yaw_peak_abs=q[..., 2:].amax((0, 2)),
                hip_roll_final_abs=q[-fn:, :, :2].mean((0, 2)), hip_yaw_final_abs=q[-fn:, :, 2:].mean((0, 2)),
                action_rate_mean=(da * dm).sum(0) / dm.sum(0).clamp(min=1), joint_acc_rms=acc_rms, joint_acc_p99=wq2(acc, 1, 0.99), joint_jerk_p99=wq2(jerk, 2, 0.99))
