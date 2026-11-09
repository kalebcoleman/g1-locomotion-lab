"""Unit tests for the scoring functions (false positives and edge cases). Run: <lab python> -I tests/test_metrics.py  (or pytest)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from g1_recovery import metrics as M

T = lambda *v: torch.tensor(v)
B = lambda *v: torch.tensor(v, dtype=torch.bool)


def test_supported_stand_requires_both_feet():
    h, up = T(0.7, 0.7, 0.7), T(0.95, 0.95, 0.95)
    f = torch.tensor([[50., 50.], [50., 0.], [0., 0.]]); z = torch.full((3, 2), 0.05); v = T(0.1, 0.1, 0.1)
    assert M.supported_stand(h, up, f, z, v).tolist() == [True, False, False]


def test_airborne_lunge_is_not_supported():            # pelvis high and upright but feet 0.35 m up and fast (the A6/A8c exploit signature)
    assert not M.supported_stand(T(0.8), T(0.97), torch.zeros(1, 2), torch.full((1, 2), 0.35), T(1.2)).item()


def test_speed_uses_3d_not_xy_only():                  # xy speed 0.1 but vertical 1.0 m/s -> 3D 1.005 -> not stable
    v3 = torch.sqrt(T(0.1) ** 2 + T(1.0) ** 2)
    assert not M.supported_stand(T(0.7), T(0.95), torch.full((1, 2), 50.), torch.full((1, 2), 0.05), v3).item()


def test_amendment3_torque_persist_stance():           # the R7m iteration-350 pattern: valid 2 s hold, but a wide straddle that drops to a kneel with torques above the real limits
    assert M.torque_ok(T(15), 500).item() and not M.torque_ok(T(16), 500).item()
    assert M.persists(T(0.90)).item() and not M.persists(T(0.10)).item()
    assert M.stance_ok(T(0.60)).item() and not M.stance_ok(T(0.88)).item()
    assert M.real_torque_limit("left_knee_joint") == 139.0 and M.real_torque_limit("torso_joint") == 88.0 and M.real_torque_limit("left_shoulder_roll_joint") == 25.0
    assert M.real_torque_limit("left_zero_joint") == float("inf")


def test_threshold_edges():
    f = torch.full((1, 2), 50.); z = torch.full((1, 2), 0.05)
    assert M.supported_stand(T(0.60), T(0.90), f, z, T(0.59)).item()
    assert not M.supported_stand(T(0.599), T(0.90), f, z, T(0.1)).item()
    assert not M.supported_stand(T(0.7), T(0.899), f, z, T(0.1)).item()
    assert not M.supported_stand(T(0.7), T(0.95), f, z, T(0.6)).item()          # strictly < 0.6


def test_plausible_flags():
    ok = B(True, True, True, True, True, True, True, True); steps = 500
    vv = torch.tensor([0, 11, 0, 0, 0, 0, 0, 0]); fl = torch.tensor([0, 0, 11, 0, 0, 0, 0, 0])
    pz = T(1., 1., 1., 2.1, 1., 1., 1., 1.); pv = torch.tensor([0, 0, 0, 0, 51, 0, 0, 0])
    assist = B(False, False, False, False, False, True, False, False)
    out = M.plausible_recovery(ok, vv, fl, pv, pz, steps, assist)
    assert out.tolist() == [True, False, False, False, False, False, True, True]
    assert not M.plausible_recovery(B(False), torch.zeros(1), torch.zeros(1), torch.zeros(1), T(0.), steps).item()   # no supported stand -> never plausible


def test_position_limit_violation_rejects():
    assert not M.plausible_recovery(B(True), torch.zeros(1), torch.zeros(1), torch.tensor([300]), T(0.5), 500).item()   # 60% of steps at limits


def test_heading_projection_rejects_sideways_motion():
    p0 = torch.zeros(2, 2); p1 = torch.tensor([[0., 4.], [3.2, 0.]]); heading = torch.tensor([[1., 0.], [1., 0.]])
    along, lat = M.heading_progress(p0, p1, heading)
    assert abs(along[0].item()) < 1e-6 and abs(along[1].item() - 3.2) < 1e-5 and abs(lat[0].item() - 4.0) < 1e-5     # 4 m sideways does NOT count


def test_heading_rotated():
    along, _ = M.heading_progress(torch.zeros(1, 2), torch.tensor([[0., 3.5]]), torch.tensor([[0., 2.]]))
    assert abs(along.item() - 3.5) < 1e-5


def test_hard_limit_rule_v21():
    out = M.hard_limits_ok(torch.tensor([0, 15, 16, 0]), T(0.0, 0.1, 0.0, 0.11), 500)     # 15/500 = 3% allowed, 16/500 not; 0.10 rad allowed, 0.11 not
    assert out.tolist() == [True, True, False, False]
    # the sit-up policy's measured behaviour (9.5% of steps, 0.22 rad) must fail; the walking reference (1.5%, 0.083) must pass
    assert M.hard_limits_ok(torch.tensor([48, 8]), T(0.22, 0.083), 500).tolist() == [False, True]


def test_posture_flags():
    f2 = torch.full((1, 2), 50.); z2 = torch.full((1, 2), 0.05); nof = torch.zeros(1, 2); hi = torch.full((1, 2), 0.3); dt = 0.02
    fl = lambda h, up, ff, fz, tr=100., v=0.1, run=0: M.posture_flags(T(h), T(up), ff, fz, T(tr), T(v), torch.tensor([run]), dt)[0].tolist()
    assert fl(0.17, 0.98, nof, z2) == [False, True, False, False, False, False, False, False]             # sitting (trunk on floor)
    assert fl(0.52, -0.97, f2, z2) == [False, False, True, False, False, False, False, False]             # forward fold is never a stand, even on its feet
    assert fl(0.40, 0.90, f2, z2, tr=0.) == [True, True, False, True, False, False, False, False]         # crouch on feet
    assert fl(0.80, 0.97, nof, hi, tr=0., v=1.2) == [True, True, False, False, False, False, False, False]  # airborne lunge: high and upright but no feet
    assert fl(0.70, 0.97, f2, z2, tr=0., run=100) == [True, True, False, True, True, True, True, True]    # 2 s supported stand


def _chain(**kw):
    d = dict(handed_over=B(True), plausible_at_handover=B(True), window_complete=B(True), fell_in_window=B(False), along_disp=T(5.0), window_s=10.0)
    d.update(kw); ok, r = M.chain_success(**d); return bool(ok.item()), int(r.item())


def test_chain_success_and_edges():
    assert _chain() == (True, 0)
    assert _chain(along_disp=T(5.0)) == (True, 0)                      # 5 m / 10 s = 0.5 m/s passes both rules
    assert _chain(along_disp=T(4.99)) == (False, 6)                    # >= 3 m but 0.499 m/s < 0.5
    assert _chain(along_disp=T(2.9)) == (False, 5)
    assert _chain(handed_over=B(False)) == (False, 1)
    assert _chain(plausible_at_handover=B(False)) == (False, 2)       # a launched recovery cannot be rescued by walking well afterwards
    assert _chain(window_complete=B(False)) == (False, 3)             # timeout
    assert _chain(fell_in_window=B(True)) == (False, 4)               # walked 5 m but fell on the way


def test_chain_displacement_alone_is_insufficient():
    assert _chain(along_disp=T(2.0)) == (False, 5)                    # the old evaluator accepted any >= 1 m Euclidean displacement


def test_reason_priority():
    assert _chain(handed_over=B(False), fell_in_window=B(True), along_disp=T(0.0)) == (False, 1)




def _trace(T=100, N=3, J=4, dt=0.02):
    spread = torch.full((T, N), 0.30); h = torch.full((T, N), 0.75)
    h[:5] = 0.10                                    # lying start: below the low gate
    hip = torch.zeros(T, N, 4); jv = torch.zeros(T, N, J); act = torch.zeros(T, N, 5)
    return spread, h, hip, jv, act, dt


def test_spread_peak_excludes_lying_start_and_reports_time():
    s, h, hip, jv, act, dt = _trace(); s[2, 0] = 1.4; s[20, 1] = 0.9; s[:, 2] = 0.2      # robot 0: spike while lying (excluded); robot 1: spike at step 20 at h=0.75
    m = M.movement_metrics(s, h, hip, jv, act, dt)
    assert abs(m["spread_peak_m"][0].item() - 0.30) < 1e-6 and abs(m["spread_peak_m"][1].item() - 0.9) < 1e-6
    assert abs(m["spread_peak_time_s"][1].item() - 20 * dt) < 1e-6 and abs(m["spread_final_m"][1].item() - 0.30) < 1e-6


def test_spread_never_risen_is_nan_and_wide_fraction():
    s, h, hip, jv, act, dt = _trace(); h[:, 0] = 0.1; s[10:30, 1] = 0.6                  # robot 0 never rises; robot 1 wide for 20 of 95 gated steps
    m = M.movement_metrics(s, h, hip, jv, act, dt)
    assert torch.isnan(m["spread_peak_m"][0]) and torch.isnan(m["spread_peak_gated_m"][0])
    assert abs(m["spread_wide_frac"][1].item() - 20 / 95) < 1e-6


def test_hip_peak_and_final_use_absolute_value_of_both_sides():
    s, h, hip, jv, act, dt = _trace(); hip[10, 0, 1] = -0.5; hip[-50:, 1, 2] = 0.2; hip[-50:, 1, 3] = -0.4   # robot 0 right roll -0.5 peak; robot 1 final yaw 0.2/-0.4
    m = M.movement_metrics(s, h, hip, jv, act, dt)
    assert abs(m["hip_roll_peak_abs"][0].item() - 0.5) < 1e-6 and abs(m["hip_yaw_final_abs"][1].item() - 0.3) < 1e-6 and m["hip_yaw_final_abs"][2].item() == 0.0


def test_acceleration_of_ramped_velocity_and_zero_jerk():
    s, h, hip, jv, act, dt = _trace(); jv[:, 0, :] = torch.arange(100)[:, None] * dt * 5.0       # d v / dt = 5 rad/s^2 on every joint
    m = M.movement_metrics(s, h, hip, jv, act, dt)
    assert abs(m["joint_acc_rms"][0].item() - 5.0) < 1e-3 and abs(m["joint_acc_p99"][0].item() - 5.0) < 1e-3 and m["joint_jerk_p99"][0].item() < 0.05 and m["joint_acc_rms"][1].item() == 0.0


def test_rising_window_limits_smoothness_and_action_rate():
    s, h, hip, jv, act, dt = _trace(); act[50:, 0, :] = torch.arange(50)[:, None] % 2 * 1.0       # chattering action only after step 50
    re = torch.tensor([40, 100, 100])
    m = M.movement_metrics(s, h, hip, jv, act, dt, rise_end=re)
    assert m["action_rate_mean"][0].item() == 0.0 and m["action_rate_mean"][1].item() == 0.0
    m2 = M.movement_metrics(s, h, hip, jv, act, dt, rise_end=torch.tensor([100, 100, 100]))
    assert m2["action_rate_mean"][0].item() > 0.4


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"): v(); print("ok", k)
