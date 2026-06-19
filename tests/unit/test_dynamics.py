"""Validation tests for the dynamics/controls modules.

Each test checks a *physical* property against an independent ground truth:
a closed-form steady state, an scipy reference integrator, the friction
break-away condition, and recovery of known motor parameters by least-squares
system identification.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from robot_testbench.motor import MotorSimulator, MotorParameters
from robot_testbench.motor.integrators import rk4_step
from robot_testbench.motor.friction import (
    FrictionParameters,
    friction_torque,
    kinetic_friction_torque,
    is_stuck,
)
from robot_testbench.analysis.system_id import (
    generate_chirp_voltage,
    collect_excitation_data,
    identify_motor_parameters,
)


def _motor_params(**kw):
    base = dict(
        inertia=0.05, damping=0.2, torque_constant=0.1, max_torque=40.0,
        max_speed=500.0, resistance=1.0, inductance=0.01,
    )
    base.update(kw)
    return MotorParameters(**base)


# --------------------------------------------------------------- integrator

def test_rk4_matches_analytic_linear_ode():
    # dx/dt = -k x  ->  x(t) = x0 * exp(-k t). RK4 should track it to high order.
    k = 3.0
    f = lambda x: -k * x
    x = np.array([1.0])
    dt, t = 0.01, 0.0
    for _ in range(100):
        x = rk4_step(f, x, dt)
        t += dt
    assert x[0] == pytest.approx(np.exp(-k * t), rel=1e-6)


def test_motor_steady_state_matches_closed_form():
    # Constant V, no Coulomb friction, no load (Ke == Kt):
    #   omega_ss = V*Kt / (R*b + Kt*Ke),  i_ss = V*b / (R*b + Kt*Ke).
    p = _motor_params()
    V = 6.0
    motor = MotorSimulator(p)
    for _ in range(20000):  # 10 s -- well past the electrical & mechanical TCs
        motor.step(0.0005, V)
    denom = p.resistance * p.damping + p.torque_constant * p.ke
    omega_ss = V * p.torque_constant / denom
    i_ss = V * p.damping / denom
    st = motor.get_state()
    assert st["motor_velocity"] == pytest.approx(omega_ss, rel=5e-3)
    assert st["current"] == pytest.approx(i_ss, rel=5e-3)


def test_rk4_motor_matches_scipy_reference():
    # Cross-check the RK4 trajectory against scipy.solve_ivp on the same smooth
    # (viscous-only) ODE.
    p = _motor_params()
    V = 6.0
    Kt, Ke, R, L, J, b = (
        p.torque_constant, p.ke, p.resistance, p.inductance, p.inertia, p.damping,
    )

    def rhs(_t, y):
        i, w = y
        return [(V - R * i - Ke * w) / L, (Kt * i - b * w) / J]

    T = 0.5
    sol = solve_ivp(rhs, [0.0, T], [0.0, 0.0], rtol=1e-9, atol=1e-12)
    i_ref, w_ref = sol.y[:, -1]

    motor = MotorSimulator(p)
    dt = 1e-4
    for _ in range(int(round(T / dt))):
        motor.step(dt, V)
    st = motor.get_state()
    assert st["current"] == pytest.approx(i_ref, rel=2e-3)
    assert st["motor_velocity"] == pytest.approx(w_ref, rel=2e-3)


# ------------------------------------------------------------------ friction

def test_friction_stiction_holds_below_breakaway():
    fp = FrictionParameters(coulomb_torque=0.5, static_torque=1.0, viscous_damping=0.1)
    # Below break-away: held static, friction exactly cancels the applied torque.
    assert is_stuck(0.0, 0.8, fp) is True
    assert friction_torque(0.0, 0.8, fp) == pytest.approx(0.8)
    # Above break-away: slips; friction caps at the static level so net torque > 0.
    assert is_stuck(0.0, 1.5, fp) is False
    assert friction_torque(0.0, 1.5, fp) == pytest.approx(1.0)


def test_friction_zero_static_never_sticks():
    # Regression: a joint with no static friction must never be 'stuck' -- this is
    # the bug that froze a frictionless motor on its first step.
    fp = FrictionParameters(coulomb_torque=0.0, static_torque=0.0, viscous_damping=0.1)
    assert is_stuck(0.0, 0.0, fp) is False
    assert is_stuck(0.0, 5.0, fp) is False


def test_friction_stribeck_hump_decays_to_coulomb():
    fp = FrictionParameters(coulomb_torque=0.5, static_torque=1.0,
                            viscous_damping=0.0, stribeck_velocity=0.1)
    f_low = kinetic_friction_torque(0.05, fp)    # near Stribeck velocity -> elevated
    f_high = kinetic_friction_torque(100.0, fp)  # high speed -> ~Coulomb
    assert f_low > fp.coulomb_torque
    assert f_high == pytest.approx(fp.coulomb_torque, rel=1e-3)


# ----------------------------------------------------------------- system ID

def test_system_id_recovers_known_parameters():
    true = _motor_params(resistance=1.2, inductance=0.02, torque_constant=0.15,
                         inertia=0.04, damping=0.25)
    motor = MotorSimulator(true)
    dt = 1e-4
    volt = generate_chirp_voltage(duration=4.0, dt=dt, amplitude=6.0, offset=5.0)
    v_log, i_log, w_log = collect_excitation_data(motor, volt, dt)
    est = identify_motor_parameters(v_log, i_log, w_log, dt)
    # ZOH-consistent identification recovers the parameters essentially exactly.
    assert est.resistance == pytest.approx(true.resistance, rel=0.02)
    assert est.inductance == pytest.approx(true.inductance, rel=0.02)
    assert est.torque_constant == pytest.approx(true.torque_constant, rel=0.02)
    assert est.inertia == pytest.approx(true.inertia, rel=0.02)
    assert est.damping == pytest.approx(true.damping, rel=0.02)
