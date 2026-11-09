"""M2: state machine and stand gate for one continuous episode  walk -> induced fall -> get-up -> stand gate -> walk.
Pure torch, simulator-free, so it is unit-tested on CPU (tests/test_fall_handover.py). Batched over environments.

States (int8):  WALK (frozen walking policy, commanded velocity) -> RECOVER (frozen recovery policy, zero command) -> WALK2 (walking resumes) -> DONE
Failures (terminal): F_NOFALL (push applied, robot never fell), F_NOSTAND (recovery window ended without the stand gate), F_REFALL (fell again while walking after hand-over),
                     F_RESET (the environment reset at any point: a reset is never credited as recovery).
Rules
- Fall detection is debounced: pelvis height < fall_h OR torso up-vector.z < fall_up for det_steps consecutive steps.
- Hysteresis: the fall thresholds (0.45 m / 0.5) are lower than the stand-gate thresholds (0.60 m / 0.90), so the band in between neither triggers a fall nor passes the gate.
  Transitions are one-way: RECOVER never returns to WALK, WALK2 never returns to RECOVER. After hand-over only the fall thresholds (debounced) count as a re-fall.
- Hand-over happens only when gate_ok is True (criteria v3 stand/hold measured over the get-up window, see window_gate).
- A reset in any non-terminal state, including WALK2 before the walking window is complete, is F_RESET."""
from __future__ import annotations
import torch
from . import metrics as M

WALK, RECOVER, WALK2, DONE, F_NOFALL, F_NOSTAND, F_REFALL, F_RESET = range(8)
NAMES = ["walk", "recover", "walk_resumed", "done", "no_fall_after_push", "no_stand_gate", "refell_after_handover", "env_reset"]
TERMINAL_FAIL = (F_NOFALL, F_NOSTAND, F_REFALL, F_RESET)


class FallHandoverFSM:
    def __init__(self, n, device, det_steps=10, fall_h=0.45, fall_up=0.5, recover_max=500, walk_steps=500, nofall_steps=150):
        self.n, self.dev = n, device
        self.det_steps, self.fall_h, self.fall_up = det_steps, fall_h, fall_up
        self.recover_max, self.walk_steps, self.nofall_steps = recover_max, walk_steps, nofall_steps
        i = lambda v=-1: torch.full((n,), v, device=device, dtype=torch.long)
        self.state = torch.zeros(n, device=device, dtype=torch.int8)
        self.cnt = i(0)                                   # consecutive steps with the fall condition (debounce counter)
        self.impulse_step = i(); self.fall_onset = i(); self.detect_step = i(); self.handover_step = i(); self.end_step = i()
        self.impulsed = torch.zeros(n, device=device, dtype=torch.bool)
        self.spontaneous = torch.zeros(n, device=device, dtype=torch.bool)    # fell before (or without) the push

    def mark_impulse(self, k, mask):
        m = mask & (self.state == WALK) & ~self.impulsed
        self.impulsed |= m; self.impulse_step = torch.where(m, torch.full_like(self.impulse_step, k), self.impulse_step)
        return m

    def step(self, k, h, up, gate_ok, reset):
        """Advance every env by one control step k. Returns a bool mask of envs that entered RECOVER at this step."""
        s = self.state; entered = torch.zeros(self.n, device=self.dev, dtype=torch.bool)
        active = ~torch.isin(s, torch.tensor([DONE, *TERMINAL_FAIL], device=self.dev, dtype=s.dtype))
        # 1. reset anywhere before DONE is a failure
        rs = reset & active
        self._to(rs, F_RESET, k); s = self.state
        fell = (h < self.fall_h) | (up < self.fall_up)
        det_state = (s == WALK) | (s == WALK2)
        self.cnt = torch.where(fell & det_state, self.cnt + 1, torch.zeros_like(self.cnt))
        onset = det_state & fell & (self.cnt == 1)
        self.fall_onset = torch.where(onset & (s == WALK), torch.full_like(self.fall_onset, k), self.fall_onset)
        # 2. WALK: debounced detection, or push without a fall
        det = (s == WALK) & (self.cnt >= self.det_steps)
        self.spontaneous |= det & ~self.impulsed
        self.detect_step = torch.where(det, torch.full_like(self.detect_step, k), self.detect_step)
        self._to(det, RECOVER, k, final=False); entered |= det; self.cnt = torch.where(det, torch.zeros_like(self.cnt), self.cnt)
        nofall = (s == WALK) & ~det & self.impulsed & ((k - self.impulse_step) >= self.nofall_steps)
        self._to(nofall, F_NOFALL, k)
        # 3. RECOVER: gate or timeout (the step that entered RECOVER cannot also pass the gate)
        rec = (s == RECOVER)
        go = rec & gate_ok
        self.handover_step = torch.where(go, torch.full_like(self.handover_step, k), self.handover_step)
        self._to(go, WALK2, k, final=False)
        to = rec & ~go & ((k - self.detect_step) >= self.recover_max)
        self._to(to, F_NOSTAND, k)
        # 4. WALK2: re-fall (fall thresholds only) or window complete
        w2 = (s == WALK2)
        self._to(w2 & (self.cnt >= self.det_steps), F_REFALL, k)
        self._to(w2 & (self.cnt < self.det_steps) & ((k - self.handover_step) >= self.walk_steps), DONE, k)
        return entered

    def _to(self, mask, new, k, final=True):
        if not mask.any(): return
        self.state = torch.where(mask, torch.full_like(self.state, new), self.state)
        if final: self.end_step = torch.where(mask, torch.full_like(self.end_step, k), self.end_step)

    def finalize(self, k):
        """End of the episode horizon: unfinished envs become failures (never silent successes)."""
        s = self.state
        self._to((s == WALK) & self.impulsed, F_NOFALL, k)       # push applied but no fall detected in time
        self._to(s == RECOVER, F_NOSTAND, k)
        # WALK without a push is the no-push control and WALK2 not yet DONE stays as is (incomplete walking window)

    def outcome(self):
        return self.state.clone()


def window_gate(run_s, hold_steps, wsteps, vel_viol_steps, flight_steps, pos_viol_steps, peak_vz, hard_viol_steps, hard_excess_max, torque_over_steps, stance_mean_hold, h, up, p: M.Params = M.P):
    """Criteria v3 stand/hold gate measured over the get-up window (steps since fall detection). All args (N,) tensors except scalars.
    Passes when: supported stand held >= hold_steps consecutively (pelvis >= 0.60 m, up >= 0.90, both feet loaded, 3D speed < 0.6) AND, over the window,
    joint-velocity-limit violations <= 2% of steps, ground-flight <= 2%, position-limit violations <= 10%, hard-limit rule (<= 3% of steps, excess <= 0.10 rad),
    torque-limit excess <= 3%, peak pelvis upward speed <= 2.0 m/s, and mean foot distance over the last hold <= 0.60 m. No assist exists in this runner.
    Not applied here: 3b self-penetration (offline screen; the recovery env runs with self-collision on) and 3c persistence (trivially met at the gate)."""
    steps = wsteps.clamp(min=1)
    supported_success = (run_s >= hold_steps) & (h >= p.stand_h) & (up >= p.stand_up)
    plaus = M.plausible_recovery(supported_success, vel_viol_steps, flight_steps, pos_viol_steps, peak_vz, steps, None)
    return plaus & M.hard_limits_ok(hard_viol_steps, hard_excess_max, steps) & M.torque_ok(torque_over_steps, steps) & M.stance_ok(stance_mean_hold)


def strict_gate(run_s, hold_steps, h, up, p: M.Params = M.P):
    """EXPLORATORY gate (not criteria v3): supported stand held >= hold_steps with pelvis >= 0.60 m and up >= 0.90, no validity clauses. Used only to show what the v3 validity
    clauses (flight, limits, torque, launch, stance) cost; never the headline."""
    return (run_s >= hold_steps) & (h >= p.stand_h) & (up >= p.stand_up)


# ---------------------------------------------------------------------------------------------------------------------
# M2b: configurable hand-over variants (evaluation only; no policy weights change). The official criteria-v3 stand result is always tracked
# separately (HandoverController.v3_first / v3_at_handover) and stays the headline; the readiness check below is a different, clearly labelled check.
READY_SPEED = 0.20       # m/s, 3D pelvis speed
READY_ANGVEL = 0.5       # rad/s, pelvis angular speed
READY_HOLD_STEPS = 50    # 1 s


def ready_check(h, up, supported, speed3d, angspeed, p: M.Params = M.P, max_speed=READY_SPEED, max_angvel=READY_ANGVEL):
    """HAND-OVER READINESS (not criteria v3): supported stance (both feet loaded, near the floor), pelvis >= 0.60 m, upright >= 0.90,
    base linear speed < max_speed and angular speed < max_angvel. Must hold for READY_HOLD_STEPS consecutive steps. No validity clauses (flight, limits, torque)."""
    return supported & (h >= p.stand_h) & (up >= p.stand_up) & (speed3d < max_speed) & (angspeed < max_angvel)


class HandoverController:
    """Decides WHEN the recovery policy hands over, and shapes the transition. Batched, pure torch.
    trigger  'v3'    : fire when the official window gate (criteria v3 stand/hold) passes (M2 behaviour)
             'ready' : fire when the readiness check has held for ready_hold steps (labelled separately from v3)
    settle   steps between the trigger and the actual switch; the recovery policy (or a hold of the current joint positions) keeps acting. If the robot stops being
             stable (supported, pelvis >= 0.60 m, up >= 0.90) during the settle, the trigger is cancelled and must fire again.
    blend    during the last `blend` steps of the settle the commanded action is ramped linearly from the policy/hold action to the default stance (action 0).
    ramp     after the switch the commanded forward velocity ramps linearly from 0 to the target over `ramp` steps.
    With trigger='v3', settle=blend=ramp=0 the behaviour equals M2."""

    def __init__(self, n, device, trigger="v3", ready_hold=READY_HOLD_STEPS, settle=0, blend=0, ramp=0):
        assert trigger in ("v3", "ready") and blend <= settle or blend == 0
        self.n, self.dev = n, device
        self.trigger, self.ready_hold, self.settle, self.blend, self.ramp = trigger, ready_hold, settle, blend, ramp
        self.trig_step = torch.full((n,), -1, device=device, dtype=torch.long)
        self.ready_run = torch.zeros(n, device=device, dtype=torch.long)
        self.v3_first = torch.full((n,), -1, device=device, dtype=torch.long)          # first step the official v3 window gate passed (while recovering)
        self.ready_first = torch.full((n,), -1, device=device, dtype=torch.long)
        self.v3_at_handover = torch.zeros(n, device=device, dtype=torch.bool)          # v3 gate true at the step the switch happened
        self.cancels = torch.zeros(n, device=device, dtype=torch.long)

    def update(self, k, in_recover, v3_ok, ready_now, stable_now):
        """Call once per step AFTER the physics step. Returns gate_ok (N,) bool: True for envs whose settle has finished at step k (hand over now)."""
        self.ready_run = torch.where(ready_now, self.ready_run + 1, torch.zeros_like(self.ready_run))
        ready_ok = self.ready_run >= self.ready_hold
        self.v3_first = torch.where(in_recover & v3_ok & (self.v3_first < 0), torch.full_like(self.v3_first, k), self.v3_first)
        self.ready_first = torch.where(in_recover & ready_ok & (self.ready_first < 0), torch.full_like(self.ready_first, k), self.ready_first)
        cond = v3_ok if self.trigger == "v3" else ready_ok
        fire = in_recover & cond & (self.trig_step < 0)
        self.trig_step = torch.where(fire, torch.full_like(self.trig_step, k), self.trig_step)
        cancel = in_recover & (self.trig_step >= 0) & ~stable_now & (self.trig_step < k)
        self.cancels += cancel.long()
        self.trig_step = torch.where(cancel, torch.full_like(self.trig_step, -1), self.trig_step)
        done = in_recover & (self.trig_step >= 0) & ((k - self.trig_step) >= self.settle)
        self.v3_at_handover = torch.where(done, v3_ok, self.v3_at_handover)
        return done

    def settling(self, in_recover):
        return in_recover & (self.trig_step >= 0)

    def blend_alpha(self, k, in_recover):
        """(N,) weight of the default stance in the commanded action for the NEXT step (0 outside the blend window)."""
        if self.blend <= 0: return torch.zeros(self.n, device=self.dev)
        s = (k + 1 - self.trig_step).float()                                       # steps since the trigger, as seen by the action taken at step k+1
        a = ((s - (self.settle - self.blend)) / self.blend).clamp(0.0, 1.0)
        return torch.where(self.settling(in_recover), a, torch.zeros_like(a))

    def command_scale(self, k, handover_step, in_walk2):
        """(N,) factor on the commanded forward velocity for the action taken at step k+1."""
        if self.ramp <= 0: return torch.ones(self.n, device=self.dev)
        sc = ((k + 1 - handover_step).float() / self.ramp).clamp(0.0, 1.0)
        return torch.where(in_walk2, sc, torch.ones_like(sc))
