"""
Cascade servo controller for the brushed-DC motor.

Topology (classic three-loop servo cascade, outer to inner):

    position error --[ P / PI ]--> velocity command
        (+ velocity feedforward)
    velocity error --[ PI ]------> torque (current) command
        (+ acceleration feedforward: tau_ff = J * a_des)
    current error  --[ PI ]------> voltage command
        (+ back-EMF feedforward: Ke * omega)

Each loop is a :class:`PIDBlock` implementing the two corrections the original
code got wrong:

1. **Back-calculation anti-windup** (exact form):

       integral += Ki * e * dt + (1 / Tt) * (u_sat - u_unsat) * dt

   Here ``integral`` is the integral term's *contribution to the output* (Ki is
   already folded in), ``u_unsat`` is the pre-saturation command and ``u_sat`` is
   ``clamp(u_unsat)``. When the output saturates, ``(u_sat - u_unsat)`` is non-
   zero and bleeds the integrator back toward a non-winding value with tracking
   time constant ``Tt``. Feedforward is included in ``u_unsat`` so the integrator
   tracks the genuinely realised output.

2. **Derivative on measurement, low-pass filtered** -- the D term differentiates
   the (negated) measurement, not the raw error, and passes it through a first-
   order filter with time constant ``tau_d``. This avoids derivative kick on
   setpoint steps and avoids amplifying measurement noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PIDBlock:
    """A single PID loop with back-calculation anti-windup and filtered D term.

    All gains are in the units appropriate to the loop. ``integral`` holds the
    integral term's contribution to the output directly (Ki folded in).
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -np.inf
    out_max: float = np.inf
    tracking_time_constant: float = 0.1  # Tt for back-calculation
    derivative_tau: float = 0.01         # first-order D filter time constant

    integral: float = 0.0
    _prev_measurement: float = 0.0
    _deriv_state: float = 0.0
    _initialized: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self._prev_measurement = 0.0
        self._deriv_state = 0.0
        self._initialized = False

    def update(self, setpoint: float, measurement: float, dt: float, feedforward: float = 0.0) -> float:
        """Compute one saturated control output.

        Args:
            setpoint: Desired value for this loop.
            measurement: Measured value for this loop.
            dt: Time step [s].
            feedforward: Feedforward term added before saturation.

        Returns:
            Saturated control output.
        """
        error = setpoint - measurement

        if not self._initialized:
            self._prev_measurement = measurement
            self._initialized = True

        # Derivative on measurement (negated), first-order low-pass filtered.
        d_term = 0.0
        if self.kd != 0.0 and dt > 0.0:
            raw = -(measurement - self._prev_measurement) / dt
            alpha = dt / (self.derivative_tau + dt)
            self._deriv_state += alpha * (raw - self._deriv_state)
            d_term = self.kd * self._deriv_state
        self._prev_measurement = measurement

        u_unsat = self.kp * error + self.integral + d_term + feedforward
        u_sat = float(np.clip(u_unsat, self.out_min, self.out_max))

        # Back-calculation anti-windup (exact form). integral carries Ki.
        if dt > 0.0:
            self.integral += (
                self.ki * error * dt
                + (1.0 / self.tracking_time_constant) * (u_sat - u_unsat) * dt
            )

        return u_sat


@dataclass
class CascadeGains:
    """Gains for the three cascade loops plus feedforward toggles."""

    # Position loop (outer): P or PI -> velocity command [rad/s]
    pos_kp: float
    pos_ki: float = 0.0
    # Velocity loop (middle): PI -> torque command [N.m]
    vel_kp: float = 0.0
    vel_ki: float = 0.0
    # Current loop (inner): PI -> voltage command [V]
    cur_kp: float = 0.0
    cur_ki: float = 0.0
    # Feedforward enables
    velocity_feedforward: bool = True
    acceleration_feedforward: bool = True
    back_emf_feedforward: bool = True


class CascadeController:
    """Three-loop (position -> velocity -> current) cascade servo controller.

    The controller needs a few plant constants for the physically-motivated
    feedforward and limit conversions: torque constant ``kt`` [N.m/A], back-EMF
    constant ``ke`` [V.s/rad], reflected inertia ``inertia`` [kg.m^2] and gear
    ratio ``gear_ratio``. Inputs to :meth:`update` are OUTPUT-side position /
    velocity and the motor winding current, matching ``MotorSimulator`` outputs.
    """

    def __init__(
        self,
        gains: CascadeGains,
        kt: float,
        ke: float,
        inertia: float,
        gear_ratio: float = 1.0,
        max_velocity: float = 50.0,        # output-side [rad/s]
        max_torque: float = 10.0,          # motor-side [N.m]
        max_voltage: float = 24.0,         # [V]
        tracking_time_constant: float = 0.05,
        derivative_tau: float = 0.005,
    ):
        self.gains = gains
        self.kt = kt
        self.ke = ke
        self.inertia = inertia            # motor-side reflected inertia
        self.gear_ratio = gear_ratio
        self.max_current = max_torque / kt

        self.pos_loop = PIDBlock(
            kp=gains.pos_kp, ki=gains.pos_ki,
            out_min=-max_velocity, out_max=max_velocity,
            tracking_time_constant=tracking_time_constant,
        )
        self.vel_loop = PIDBlock(
            kp=gains.vel_kp, ki=gains.vel_ki,
            out_min=-max_torque, out_max=max_torque,
            tracking_time_constant=tracking_time_constant,
        )
        self.cur_loop = PIDBlock(
            kp=gains.cur_kp, ki=gains.cur_ki,
            out_min=-max_voltage, out_max=max_voltage,
            tracking_time_constant=tracking_time_constant,
        )

        # Setpoints / trajectory references.
        self.position_setpoint = 0.0
        self.velocity_setpoint = 0.0        # velocity feedforward (output side)
        self.acceleration_setpoint = 0.0    # accel feedforward (output side)

        self._last = {"vel_cmd": 0.0, "torque_cmd": 0.0, "current_cmd": 0.0}

    def reset(self) -> None:
        self.pos_loop.reset()
        self.vel_loop.reset()
        self.cur_loop.reset()
        self._last = {"vel_cmd": 0.0, "torque_cmd": 0.0, "current_cmd": 0.0}

    def set_target(self, position: float, velocity: float = 0.0, acceleration: float = 0.0) -> None:
        """Set the position setpoint and optional velocity/accel feedforward refs."""
        self.position_setpoint = position
        self.velocity_setpoint = velocity
        self.acceleration_setpoint = acceleration

    def update(self, position: float, velocity: float, current: float, omega_motor: float, dt: float) -> float:
        """Run all three loops and return the commanded terminal voltage [V].

        Args:
            position: Output-side angle [rad].
            velocity: Output-side angular velocity [rad/s].
            current: Motor winding current [A].
            omega_motor: Motor-shaft speed [rad/s] (for back-EMF feedforward).
            dt: Time step [s].
        """
        g = self.gains

        # --- Outer position loop -> velocity command (output side) ---
        vel_ff = self.velocity_setpoint if g.velocity_feedforward else 0.0
        vel_cmd = self.pos_loop.update(self.position_setpoint, position, dt, feedforward=vel_ff)

        # --- Middle velocity loop -> torque command (motor side) ---
        # Acceleration feedforward: tau = J * a_des, referred to the motor shaft.
        # a_des is output-side; motor-side accel = a_des * N, so tau_ff = J * a_des * N.
        accel_ff = (
            self.inertia * self.acceleration_setpoint * self.gear_ratio
            if g.acceleration_feedforward
            else 0.0
        )
        torque_cmd = self.vel_loop.update(vel_cmd, velocity, dt, feedforward=accel_ff)

        # --- Inner current loop -> voltage command ---
        current_cmd = torque_cmd / self.kt
        current_cmd = float(np.clip(current_cmd, -self.max_current, self.max_current))
        # Back-EMF feedforward linearises the electrical plant: V_ff = Ke * omega_m.
        emf_ff = self.ke * omega_motor if g.back_emf_feedforward else 0.0
        voltage = self.cur_loop.update(current_cmd, current, dt, feedforward=emf_ff)

        self._last = {"vel_cmd": vel_cmd, "torque_cmd": torque_cmd, "current_cmd": current_cmd}
        return voltage

    def get_internal(self) -> dict:
        """Expose the last intermediate commands for debugging / plotting."""
        return dict(self._last)
