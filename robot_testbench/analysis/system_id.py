"""
Least-squares system identification of brushed-DC motor parameters.

Given a logged excitation -- the per-step (zero-order-hold) terminal voltage V
and the motor-side current i and shaft speed omega sampled at the interval
endpoints -- recover (R, L, Kt, J, b). The motor is linear in its parameters, so
identification splits into two ordinary least-squares regressions.

To stay consistent with how the plant is actually driven (a fixed-rate digital
controller holding V constant over each step), each governing equation is
**integrated over one ZOH interval** [t_k, t_{k+1}] rather than differentiated
pointwise. Over that interval V is exactly constant, and the integral of each
derivative term telescopes to the endpoint change, which removes the bias that
pointwise central differences incur across the voltage steps:

  Electrical:  V_k = R * i_avg + Ke * w_avg + L * (i_{k+1} - i_k) / dt
  Mechanical:  Kt * i_avg = J * (w_{k+1} - w_k) / dt + b * w_avg

where ``i_avg = (i_k + i_{k+1})/2`` and ``w_avg = (w_k + w_{k+1})/2`` are the
trapezoidal interval means. The electrical fit yields [R, Ke, L]; for an ideal
SI brushed-DC motor Ke == Kt, so the mechanical fit (with Kt known) yields [J, b].
With clean, finely-sampled data this recovers the true parameters to within the
O(dt^2) trapezoidal error -- effectively exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotorParamEstimate:
    """Recovered motor parameters."""

    resistance: float       # R [Ohm]
    inductance: float       # L [H]
    torque_constant: float  # Kt [N.m/A] (== Ke)
    inertia: float          # J [kg.m^2]
    damping: float          # b [N.m.s/rad]


def generate_chirp_voltage(
    duration: float,
    dt: float,
    amplitude: float = 6.0,
    f0: float = 0.5,
    f1: float = 40.0,
    offset: float = 4.0,
) -> np.ndarray:
    """Linear-sweep (chirp) voltage plus DC offset for rich excitation.

    The offset keeps the motor spinning (so omega, di/dt and domega/dt are all
    excited) while the sweep injects energy across a band of frequencies, which
    makes the regression well-conditioned.
    """
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    k = (f1 - f0) / duration
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return offset + amplitude * np.sin(phase)


def collect_excitation_data(motor, voltages: np.ndarray, dt: float):
    """Drive ``motor`` with a ZOH voltage sequence and log the state.

    Returns ``(voltages, i, omega)`` where ``voltages`` is the per-interval
    applied voltage (length ``n``) and ``i`` / ``omega`` are the motor-side
    winding current and shaft speed sampled at the interval *endpoints* (length
    ``n + 1``: the initial state plus one sample after each step). Endpoint
    logging is what lets :func:`identify_motor_parameters` integrate each ZOH
    interval exactly.
    """
    motor.reset()
    n = len(voltages)
    i_log = np.empty(n + 1)
    w_log = np.empty(n + 1)
    i_log[0] = motor.current
    w_log[0] = motor.get_state()["motor_velocity"]
    for k in range(n):
        motor.step(dt, float(voltages[k]))
        i_log[k + 1] = motor.current
        w_log[k + 1] = motor.get_state()["motor_velocity"]
    return np.asarray(voltages, dtype=float), i_log, w_log


def identify_motor_parameters(
    voltage: np.ndarray,
    current: np.ndarray,
    omega: np.ndarray,
    dt: float,
) -> MotorParamEstimate:
    """Recover (R, L, Kt, J, b) from a ZOH excitation log via least squares.

    Args:
        voltage: Per-interval (zero-order-hold) terminal voltage [V], length ``n``.
        current: Motor winding current at interval endpoints [A], length ``n + 1``.
        omega: Motor-shaft speed at interval endpoints [rad/s], length ``n + 1``.
        dt: Sample period [s].

    Returns:
        :class:`MotorParamEstimate`.
    """
    V = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    w = np.asarray(omega, dtype=float)
    if i.shape != w.shape:
        raise ValueError("current and omega must have the same length")
    if i.size != V.size + 1:
        raise ValueError(
            "current/omega must hold interval endpoints (length len(voltage)+1); "
            "use collect_excitation_data() to log them"
        )

    i0, i1 = i[:-1], i[1:]
    w0, w1 = w[:-1], w[1:]
    i_avg = 0.5 * (i0 + i1)
    w_avg = 0.5 * (w0 + w1)
    di_dt = (i1 - i0) / dt
    dw_dt = (w1 - w0) / dt

    # --- Electrical fit: V = R*i_avg + Ke*w_avg + L*di/dt ---
    A_elec = np.column_stack([i_avg, w_avg, di_dt])
    coef_elec, *_ = np.linalg.lstsq(A_elec, V, rcond=None)
    R, Ke, L = coef_elec
    Kt = Ke  # SI brushed-DC: Ke == Kt

    # --- Mechanical fit: Kt*i_avg = J*dw/dt + b*w_avg ---
    A_mech = np.column_stack([dw_dt, w_avg])
    coef_mech, *_ = np.linalg.lstsq(A_mech, Kt * i_avg, rcond=None)
    J, b = coef_mech

    return MotorParamEstimate(
        resistance=float(R),
        inductance=float(L),
        torque_constant=float(Kt),
        inertia=float(J),
        damping=float(b),
    )
