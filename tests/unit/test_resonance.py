"""Validation tests for the two-mass flexible-joint resonance model and input
shaping.

Each test checks a *physical* property against an independent ground truth:

  * the frequency response peaks at the analytic resonance and dips at the
    analytic anti-resonance,
  * a rest-to-rest move, once input-shaped, leaves markedly less residual
    vibration on the load than the unshaped move,
  * the ZV/ZVD shaper impulses have the textbook amplitudes and timing,
  * the time-domain RK4 plant agrees with an independent scipy reference
    integrator.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from robot_testbench.resonance import (
    FlexibleJointParameters,
    FlexibleJointSimulator,
    resonance_modes,
    transfer_function,
    frequency_response,
    frequency_response_magnitude,
    compliance_ratio,
    rigid_body_response,
    simulate_load_response,
    zv_shaper,
    zvd_shaper,
    build_shaper,
    apply_input_shaper,
    residual_vibration,
)


def _plant(**kw) -> FlexibleJointParameters:
    base = dict(motor_inertia=0.05, load_inertia=0.02, stiffness=400.0, damping=0.05)
    base.update(kw)
    return FlexibleJointParameters(**base)


# --------------------------------------------------------------- modal analysis

def test_resonance_modes_match_closed_form():
    p = _plant(damping=0.0)
    m = resonance_modes(p)
    Jm, Jl, k = p.motor_inertia, p.load_inertia, p.stiffness
    assert m.omega_res == pytest.approx(np.sqrt(k * (Jm + Jl) / (Jm * Jl)))
    assert m.omega_antires == pytest.approx(np.sqrt(k / Jl))
    # Anti-resonance is always below resonance for a two-mass plant.
    assert m.omega_antires < m.omega_res
    assert m.f_res == pytest.approx(m.omega_res / (2.0 * np.pi))
    # Undamped -> zeta == 0 and damped freq equals the natural freq.
    assert m.zeta == pytest.approx(0.0)
    assert m.omega_damped == pytest.approx(m.omega_res)


def test_damping_ratio_closed_form():
    p = _plant(damping=0.05)
    m = resonance_modes(p)
    Jm, Jl, k, c = p.motor_inertia, p.load_inertia, p.stiffness, p.damping
    zeta_expected = c * (Jm + Jl) / (2.0 * np.sqrt(k * (Jm + Jl) * Jm * Jl))
    assert m.zeta == pytest.approx(zeta_expected)
    assert 0.0 < m.zeta < 1.0  # underdamped resonant pair


# ------------------------------------------------------------- frequency resp.

def test_frf_peak_at_resonance():
    # The load-side compliance ratio must peak within a few % of omega_res.
    p = _plant()
    m = resonance_modes(p)
    omega = np.linspace(1.0, 3.0 * m.omega_res, 60000)
    mag = compliance_ratio(p, omega, output="load")
    omega_peak = omega[np.argmax(mag)]
    assert omega_peak == pytest.approx(m.omega_res, rel=0.03)
    # A lightly-damped resonance is a genuine amplification, not a shoulder.
    assert mag.max() > 5.0


def test_frf_antiresonance_dip():
    # The collocated (motor-side) response must dip near omega_antires.
    p = _plant()
    m = resonance_modes(p)
    omega = np.linspace(1.0, 3.0 * m.omega_res, 60000)
    mag = compliance_ratio(p, omega, output="motor")
    band = (omega > 0.6 * m.omega_antires) & (omega < 1.4 * m.omega_antires)
    omega_dip = omega[band][np.argmin(mag[band])]
    assert omega_dip == pytest.approx(m.omega_antires, rel=0.03)
    # At the notch the collocated response is strongly attenuated vs the DC gain.
    assert mag[band].min() < 0.5


def test_load_frf_has_no_antiresonance():
    # The non-collocated (load) numerator is (c s + k): no finite zero pair, so
    # no anti-resonance notch -- only the resonance peak. Sanity-check that the
    # load magnitude does NOT collapse near omega_antires the way the motor does.
    p = _plant()
    m = resonance_modes(p)
    omega = np.array([m.omega_antires])
    load_mag = compliance_ratio(p, omega, output="load")[0]
    motor_mag = compliance_ratio(p, omega, output="motor")[0]
    assert load_mag > motor_mag  # load is NOT notched here; motor is.


# ---------------------------------------------------------------- time domain

def test_rk4_plant_matches_scipy_reference():
    # Cross-check the RK4 two-mass trajectory against scipy.solve_ivp on the same
    # linear ODE under a constant torque.
    p = _plant()
    Jm, Jl, k, c = p.motor_inertia, p.load_inertia, p.stiffness, p.damping
    tau = 2.0

    def rhs(_t, y):
        tm, wm, tl, wl = y
        tau_shaft = k * (tm - tl) + c * (wm - wl)
        return [wm, (tau - tau_shaft) / Jm, wl, tau_shaft / Jl]

    T = 0.3
    sol = solve_ivp(rhs, [0.0, T], [0.0, 0.0, 0.0, 0.0], rtol=1e-10, atol=1e-12)
    ref = sol.y[:, -1]

    sim = FlexibleJointSimulator(p)
    dt = 1e-4
    for _ in range(int(round(T / dt))):
        sim.step(dt, tau)
    st = sim.get_state()
    assert st["motor_angle"] == pytest.approx(ref[0], rel=2e-3, abs=1e-6)
    assert st["load_angle"] == pytest.approx(ref[2], rel=2e-3, abs=1e-6)


def test_unshaped_move_rings_at_resonance():
    # A rest-to-rest bang-bang torque move excites the resonance: the load tail
    # should still be oscillating appreciably about its final angle.
    p = _plant()
    dt = 1e-4
    n = int(2.0 / dt)
    cmd = np.zeros(n)
    ta = int(0.05 / dt)
    cmd[:ta] = 1.0
    cmd[ta : 2 * ta] = -1.0  # net-zero impulse -> ends at rest at a new angle
    _, theta_l = simulate_load_response(p, cmd, dt)
    assert residual_vibration(theta_l, 0.4) > 1e-4  # clearly ringing


# -------------------------------------------------------------- input shaping

def test_zv_shaper_amplitudes_and_timing():
    p = _plant()
    m = resonance_modes(p)
    sh = zv_shaper(m.omega_res, m.zeta)
    # Two impulses, amplitudes sum to 1 (unity DC gain).
    assert sh.amplitudes.size == 2
    assert sh.amplitudes.sum() == pytest.approx(1.0)
    # Timing: second impulse at T_d / 2 = pi / omega_d.
    omega_d = m.omega_res * np.sqrt(1.0 - m.zeta ** 2)
    assert sh.times[0] == 0.0
    assert sh.times[1] == pytest.approx(np.pi / omega_d)
    # Amplitudes match A1=1/(1+K), A2=K/(1+K).
    K = np.exp(-m.zeta * np.pi / np.sqrt(1.0 - m.zeta ** 2))
    assert sh.amplitudes[0] == pytest.approx(1.0 / (1.0 + K))
    assert sh.amplitudes[1] == pytest.approx(K / (1.0 + K))


def test_zvd_shaper_amplitudes_and_timing():
    p = _plant()
    m = resonance_modes(p)
    sh = zvd_shaper(m.omega_res, m.zeta)
    assert sh.amplitudes.size == 3
    assert sh.amplitudes.sum() == pytest.approx(1.0)
    omega_d = m.omega_res * np.sqrt(1.0 - m.zeta ** 2)
    Td = 2.0 * np.pi / omega_d
    assert sh.times == pytest.approx([0.0, Td / 2.0, Td])
    K = np.exp(-m.zeta * np.pi / np.sqrt(1.0 - m.zeta ** 2))
    D = 1.0 + 2.0 * K + K ** 2
    assert sh.amplitudes == pytest.approx([1.0 / D, 2.0 * K / D, K ** 2 / D])


def test_shaper_preserves_dc_gain():
    # Convolving a constant command must leave the steady-state value unchanged
    # (the impulse weights sum to 1), so shaping never changes where the move
    # ends up -- only how it gets there.
    p = _plant()
    m = resonance_modes(p)
    cmd = np.ones(20000)
    shaped = apply_input_shaper(cmd, 1e-4, m.omega_res, m.zeta, kind="ZV")
    assert shaped[-1] == pytest.approx(1.0, abs=1e-9)


def test_input_shaping_suppresses_residual_vibration():
    # THE headline result: a rest-to-rest move, when ZV-shaped at the design
    # frequency, leaves far less residual load vibration than unshaped.
    p = _plant()
    m = resonance_modes(p)
    dt = 1e-4
    n = int(2.0 / dt)
    cmd = np.zeros(n)
    ta = int(0.05 / dt)
    cmd[:ta] = 1.0
    cmd[ta : 2 * ta] = -1.0

    _, theta_l_unshaped = simulate_load_response(p, cmd, dt)
    shaped = apply_input_shaper(cmd, dt, m.omega_res, m.zeta, kind="ZV")
    _, theta_l_shaped = simulate_load_response(p, shaped, dt)

    rv_unshaped = residual_vibration(theta_l_unshaped, 0.4)
    rv_shaped = residual_vibration(theta_l_shaped, 0.4)

    # Shaped residual must be a small fraction of the unshaped ringing.
    assert rv_shaped < 0.25 * rv_unshaped
    # And both moves still arrive at the same final load angle.
    final_unshaped = theta_l_unshaped[int(0.8 * n):].mean()
    final_shaped = theta_l_shaped[int(0.8 * n):].mean()
    assert final_shaped == pytest.approx(final_unshaped, rel=0.02)


def test_zvd_more_robust_than_zv_to_frequency_error():
    # ZVD is designed to tolerate a mis-estimated resonance. With a deliberate
    # +12% frequency error, ZVD should leave less residual vibration than ZV.
    p = _plant()
    m = resonance_modes(p)
    dt = 1e-4
    n = int(2.0 / dt)
    cmd = np.zeros(n)
    ta = int(0.05 / dt)
    cmd[:ta] = 1.0
    cmd[ta : 2 * ta] = -1.0

    omega_wrong = 1.12 * m.omega_res  # mis-estimated design frequency
    zv = apply_input_shaper(cmd, dt, omega_wrong, m.zeta, kind="ZV")
    zvd = apply_input_shaper(cmd, dt, omega_wrong, m.zeta, kind="ZVD")
    _, theta_zv = simulate_load_response(p, zv, dt)
    _, theta_zvd = simulate_load_response(p, zvd, dt)
    assert residual_vibration(theta_zvd, 0.4) < residual_vibration(theta_zv, 0.4)


def test_build_shaper_dispatch():
    p = _plant()
    m = resonance_modes(p)
    assert build_shaper(m.omega_res, m.zeta, "ZV").amplitudes.size == 2
    assert build_shaper(m.omega_res, m.zeta, "zvd").amplitudes.size == 3
    with pytest.raises(ValueError):
        build_shaper(m.omega_res, m.zeta, "bogus")
