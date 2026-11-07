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


CHAIN_REASONS = ["success", "never_handed_over", "implausible_recovery", "timeout_window_incomplete", "fell_after_handover", "insufficient_progress_along_heading", "too_slow"]
