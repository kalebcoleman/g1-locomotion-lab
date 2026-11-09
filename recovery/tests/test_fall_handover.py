"""CPU unit tests for the M2 state machine and stand gate. Run: <lab python> -I tests/test_fall_handover.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from g1_recovery import fall_handover as F

DEV = "cpu"
def mk(n=1, **kw):
    d = dict(det_steps=3, recover_max=20, walk_steps=10, nofall_steps=5); d.update(kw)
    return F.FallHandoverFSM(n, DEV, **d)
def run(fsm, seq, start=0):
    """seq: list of (h, up, gate_ok, reset) tuples (python scalars or lists per env); returns list of states after each step."""
    out = []
    for i, (h, up, g, r) in enumerate(seq):
        T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=DEV).reshape(-1).expand(fsm.n).clone()
        fsm.step(start + i, T(h), T(up), T(g, torch.bool), T(r, torch.bool)); out.append(int(fsm.state[0]))
    return out
UP = (0.75, 0.98); LOW = (0.2, 0.2)
ok = lambda hu, g=False, r=False: (hu[0], hu[1], g, r)


def test_single_dip_does_not_trigger_detection():
    f = mk(); seq = [ok(UP)] * 3 + [ok(LOW)] * 2 + [ok(UP)] * 3 + [ok(LOW)] * 2 + [ok(UP)] * 4       # dips shorter than det_steps=3
    assert set(run(f, seq)) == {F.WALK}


def test_debounced_detection_and_latency():
    f = mk(); f.mark_impulse(2, torch.tensor([True]))
    st = run(f, [ok(UP)] * 3 + [ok(LOW)] * 4)
    assert st[:5] == [F.WALK] * 5 and st[5] == F.RECOVER           # fell condition from step 3, third consecutive step is 5
    assert int(f.detect_step[0]) == 5 and int(f.fall_onset[0]) == 3 and int(f.detect_step[0] - f.impulse_step[0]) == 3
    assert not bool(f.spontaneous[0])


def test_tilt_alone_triggers_detection():
    f = mk(); st = run(f, [ok(UP)] * 2 + [(0.7, 0.3, False, False)] * 3)
    assert st[-1] == F.RECOVER


def test_push_without_fall_is_a_failure_and_control_is_not():
    f = mk(2); f.mark_impulse(0, torch.tensor([True, False]))
    for k in range(8):
        f.step(k, torch.full((2,), 0.75), torch.full((2,), 0.98), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    assert int(f.state[0]) == F.F_NOFALL and int(f.state[1]) == F.WALK                    # control (no push) stays WALK


def test_finalize_marks_unfinished_push_as_failure():
    f = mk(); f.mark_impulse(0, torch.tensor([True])); run(f, [ok(UP)] * 2); f.finalize(2)
    assert int(f.state[0]) == F.F_NOFALL


def test_no_handover_without_gate_and_timeout():
    f = mk(); st = run(f, [ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=False)] * 25)
    assert st[4] == F.RECOVER and st[10] == F.RECOVER and st[-1] == F.F_NOSTAND                # upright for many steps but the gate never passed


def test_handover_only_when_gate_ok_and_not_on_the_entering_step():
    f = mk(); st = run(f, [ok(UP)] * 2 + [ok(LOW)] * 2 + [ok(LOW, g=True)])                  # det_steps=3: the third low step (index 4) enters RECOVER with gate_ok already True
    assert st[3] == F.WALK and st[4] == F.RECOVER and int(f.handover_step[0]) == -1
    f2 = mk(); st2 = run(f2, [ok(UP)] * 2 + [ok(LOW, g=True)] * 3)
    assert st2[3] == F.WALK, "debounce not yet met"
    assert st2[4] == F.RECOVER, "entering step must not also pass the gate"
    f3 = mk(); st3 = run(f3, [ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=True)])
    assert st3[-1] == F.WALK2 and int(f3.handover_step[0]) == 5


def test_hysteresis_band_and_one_way_transitions():
    f = mk(); run(f, [ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=True)])                         # now WALK2, handover at step 5
    band = (0.52, 0.8)                                                                       # between fall (0.45) and stand (0.60) thresholds
    st = run(f, [ok(band)] * 6, start=6)
    assert set(st) == {F.WALK2}, "band must neither re-trigger a fall nor return to RECOVER"
    st = run(f, [(0.2, 0.98, False, False)] * 2, start=12)
    assert st[-1] == F.WALK2 and f.cnt[0] == 2                                              # debounce: 2 < 3 steps
    st = run(f, [(0.2, 0.98, False, False)] * 2, start=14)
    assert st[0] == F.F_REFALL and F.RECOVER not in st                                      # never back to RECOVER


def test_walk_window_completes_to_done():
    f = mk(walk_steps=6); run(f, [ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=True)])
    st = run(f, [ok(UP)] * 10, start=6)
    assert st[4] == F.WALK2 and st[5] == F.DONE and int(f.handover_step[0]) == 5


def test_reset_counts_as_failure_in_every_state():
    for prefix, label in (([ok(UP)] * 2, "walk"), ([ok(UP)] * 2 + [ok(LOW)] * 3, "recover"), ([ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=True)], "walk2")):
        f = mk(); run(f, prefix); st = run(f, [ok(UP, r=True)] + [ok(UP)] * 3, start=len(prefix))
        assert st[0] == F.F_RESET and set(st) == {F.F_RESET}, label


def test_reset_beats_gate_on_the_same_step_and_never_credits_recovery():
    f = mk(); run(f, [ok(UP)] * 2 + [ok(LOW)] * 3); st = run(f, [(0.75, 0.98, True, True)], start=5)
    assert st[0] == F.F_RESET and int(f.handover_step[0]) == -1


def test_reset_after_done_does_not_change_the_result():
    f = mk(walk_steps=2); run(f, [ok(UP)] * 2 + [ok(LOW)] * 3 + [ok(UP, g=True)]); run(f, [ok(UP)] * 4, start=6); assert int(f.state[0]) == F.DONE
    run(f, [ok(UP, r=True)], start=10); assert int(f.state[0]) == F.DONE


def test_spontaneous_fall_without_push_is_flagged():
    f = mk(); run(f, [ok(UP)] * 2 + [ok(LOW)] * 3)
    assert int(f.state[0]) == F.RECOVER and bool(f.spontaneous[0])


def _gate(**over):
    n = 1; z = lambda v, d=torch.float32: torch.tensor([v], dtype=d)
    a = dict(run_s=z(120, torch.int32), hold_steps=100, wsteps=z(300, torch.int32), vel_viol_steps=z(0, torch.int32), flight_steps=z(0, torch.int32), pos_viol_steps=z(0, torch.int32), peak_vz=z(0.8),
             hard_viol_steps=z(0, torch.int32), hard_excess_max=z(0.0), torque_over_steps=z(0, torch.int32), stance_mean_hold=z(0.4), h=z(0.72), up=z(0.98))
    a.update({k: (torch.tensor([v], dtype=a[k].dtype) if torch.is_tensor(a[k]) else v) for k, v in over.items()})
    return bool(F.window_gate(**a)[0])


def test_gate_passes_clean_window_and_each_clause_can_fail_it():
    assert _gate()
    assert not _gate(run_s=99), "hold shorter than 2 s"
    assert not _gate(h=0.55), "pelvis below 0.60 m"
    assert not _gate(up=0.85), "tilt too large"
    assert not _gate(vel_viol_steps=7), "joint velocity limit on > 2% of steps (7/300)"
    assert not _gate(flight_steps=7), "ground flight on > 2% of steps"
    assert not _gate(pos_viol_steps=31), "position limit on > 10% of steps"
    assert not _gate(hard_excess_max=0.2), "hard-limit excursion > 0.10 rad"
    assert not _gate(hard_viol_steps=10), "hard limit on > 3% of steps"
    assert not _gate(torque_over_steps=10), "torque limit on > 3% of steps"
    assert not _gate(peak_vz=2.5), "launch"
    assert not _gate(stance_mean_hold=0.7), "wide straddle"


def test_strict_gate_is_weaker_than_v3_gate():
    z = lambda v, d=torch.float32: torch.tensor([v], dtype=d)
    assert bool(F.strict_gate(z(120, torch.int32), 100, z(0.72), z(0.98))[0]) and not bool(F.strict_gate(z(99, torch.int32), 100, z(0.72), z(0.98))[0])
    assert not bool(F.strict_gate(z(120, torch.int32), 100, z(0.5), z(0.98))[0])
    assert not _gate(flight_steps=30) and bool(F.strict_gate(z(120, torch.int32), 100, z(0.72), z(0.98))[0]), "a window the v3 gate rejects for flight still passes the strict stand gate"


if __name__ == "__main__":
    n = 0
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k); n += 1
    print(n, "passed")
