"""Behavioural tests for the brushed-DC MotorSimulator (RK4 + friction + gearbox)."""

import numpy as np
import pytest

from robot_testbench.motor import MotorSimulator, MotorParameters


def _params(**kw):
    base = dict(
        inertia=0.05, damping=0.2, torque_constant=0.1, max_torque=40.0,
        max_speed=500.0, resistance=1.0, inductance=0.01,
    )
    base.update(kw)
    return MotorParameters(**base)


def test_motor_spins_up_under_positive_voltage():
    motor = MotorSimulator(_params())
    for _ in range(500):
        pos, vel, cur = motor.step(0.001, 5.0)
    assert vel > 0.0
    assert pos > 0.0
    assert cur > 0.0


def test_motor_holds_still_under_zero_voltage():
    motor = MotorSimulator(_params())
    for _ in range(500):
        pos, vel, cur = motor.step(0.001, 0.0)
    assert vel == pytest.approx(0.0, abs=1e-9)
    assert pos == pytest.approx(0.0, abs=1e-9)
    assert cur == pytest.approx(0.0, abs=1e-9)


def test_motor_reverses_under_negative_voltage():
    motor = MotorSimulator(_params())
    for _ in range(500):
        pos, vel, cur = motor.step(0.001, -5.0)
    assert vel < 0.0
    assert cur < 0.0


def test_motor_is_deterministic():
    # No wall-clock dependence: identical inputs -> bit-identical outputs.
    a = MotorSimulator(_params())
    b = MotorSimulator(_params())
    da = np.array([a.step(0.001, 6.0) for _ in range(500)])
    db = np.array([b.step(0.001, 6.0) for _ in range(500)])
    assert np.array_equal(da, db)


def test_gearbox_reflects_inertia_and_back_emf():
    # Reflected inertia scales as 1/N^2; the output-side back-EMF constant as Ke*N.
    p = _params(gear_ratio=10.0)
    motor = MotorSimulator(p)
    assert motor.reflected_inertia(2.0) == pytest.approx(2.0 / 10.0 ** 2)
    assert motor.effective_output_back_emf_constant() == pytest.approx(p.ke * 10.0)


def test_higher_voltage_gives_higher_steady_speed():
    speeds = []
    for V in (3.0, 6.0, 9.0):
        m = MotorSimulator(_params())
        for _ in range(5000):
            m.step(0.001, V)
        speeds.append(m.velocity)
    assert speeds[0] < speeds[1] < speeds[2]
