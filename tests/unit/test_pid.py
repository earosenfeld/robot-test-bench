"""Tests for the cascade servo controller and its PID block.

Covers the two things the original PID got wrong -- back-calculation anti-windup
and derivative-on-measurement -- plus an end-to-end closed-loop position move.
"""

import numpy as np
import pytest

from robot_testbench.control.cascade import PIDBlock, CascadeGains, CascadeController
from robot_testbench.motor import MotorSimulator, MotorParameters


def test_pidblock_back_calculation_limits_windup():
    sat = dict(ki=50.0, out_min=-1.0, out_max=1.0)
    aw = PIDBlock(kp=1.0, tracking_time_constant=0.05, **sat)
    no_aw = PIDBlock(kp=1.0, tracking_time_constant=1e9, **sat)  # back-calc ~disabled
    for _ in range(300):
        aw.update(setpoint=10.0, measurement=0.0, dt=0.01)
        no_aw.update(setpoint=10.0, measurement=0.0, dt=0.01)
    # With back-calculation the integrator stays bounded; without it, it runs away.
    assert abs(aw.integral) < abs(no_aw.integral) / 50.0
    assert abs(aw.integral) < 50.0


def test_pidblock_no_derivative_kick_on_setpoint_step():
    blk = PIDBlock(kp=0.0, ki=0.0, kd=1.0, derivative_tau=0.01)
    blk.update(setpoint=0.0, measurement=0.0, dt=0.01)
    # Setpoint jumps but measurement is unchanged: D-on-measurement -> no kick.
    u = blk.update(setpoint=100.0, measurement=0.0, dt=0.01)
    assert u == pytest.approx(0.0, abs=1e-9)


def test_pidblock_derivative_opposes_measurement_rise():
    blk = PIDBlock(kp=0.0, kd=1.0, derivative_tau=1e-6)  # near-unfiltered
    blk.update(setpoint=0.0, measurement=0.0, dt=0.01)
    u = blk.update(setpoint=0.0, measurement=1.0, dt=0.01)
    assert u < 0.0  # derivative resists the rising measurement


def _plant():
    return MotorParameters(
        inertia=0.05, damping=0.2, torque_constant=0.1, max_torque=4.0,
        max_speed=50.0, resistance=1.0, inductance=0.001,
    )


def test_cascade_position_move_converges():
    p = _plant()
    gains = CascadeGains(
        pos_kp=25.0, pos_ki=0.0,
        vel_kp=0.6, vel_ki=6.0,
        cur_kp=6.0, cur_ki=1200.0,
    )
    ctrl = CascadeController(
        gains, kt=p.torque_constant, ke=p.ke, inertia=p.inertia,
        max_velocity=50.0, max_torque=4.0, max_voltage=24.0,
    )
    ctrl.set_target(1.0)
    motor = MotorSimulator(p)
    dt = 0.0005
    pos = vel = cur = omega_m = 0.0
    for _ in range(8000):  # 4 s
        voltage = ctrl.update(pos, vel, cur, omega_m, dt)
        pos, vel, cur = motor.step(dt, voltage)
        omega_m = motor.get_state()["motor_velocity"]
    assert pos == pytest.approx(1.0, abs=0.05)  # reaches the 1.0 rad setpoint
    assert abs(vel) < 0.5                        # and settles
