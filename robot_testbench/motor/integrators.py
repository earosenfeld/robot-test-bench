"""
Fixed-step numerical integrators for the motor state ODEs.

The motor simulator previously used forward (explicit) Euler, which is only
first-order accurate and goes unstable for the stiff electrical sub-system
(small L/R time constant) unless dt is tiny. This module provides a classic
fourth-order Runge-Kutta (RK4) step, which is fourth-order accurate and has a
much larger stable region, so the same dt gives dramatically lower error.

The state derivative is supplied as a callable ``f(state) -> dstate`` where both
are 1-D numpy arrays. Inputs (voltage, load torque) are captured by the caller's
closure and held constant over the step (zero-order hold), which is the standard
assumption for a fixed-rate digital controller driving the plant.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

StateDeriv = Callable[[np.ndarray], np.ndarray]


def rk4_step(f: StateDeriv, state: np.ndarray, dt: float) -> np.ndarray:
    """Advance ``state`` by one fixed RK4 step of size ``dt``.

    Args:
        f: Derivative function, ``f(state) -> dstate/dt``. Must be pure (no
            hidden time dependence); inputs are held constant over the step.
        state: Current state vector.
        dt: Fixed time step [s].

    Returns:
        New state vector after one RK4 step.
    """
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
