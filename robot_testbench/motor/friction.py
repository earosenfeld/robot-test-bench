"""
Friction models for the actuator drivetrain.

Implements the standard Coulomb + viscous + Stribeck friction curve used in
servo-drive identification work, plus correct zero-velocity (stiction) handling
so that a motor at rest does not chatter on ``sign(omega)``.

Curve (for omega != 0):

    tau_f(omega) = ( tau_c + (tau_s - tau_c) * exp(-(omega / omega_s)**2) ) * sign(omega)
                   + b * omega

where
    tau_c   : Coulomb (kinetic) friction torque magnitude [N.m]
    tau_s   : static / break-away friction torque magnitude [N.m]  (tau_s >= tau_c)
    omega_s : Stribeck velocity [rad/s] -- sets how fast friction decays from
              tau_s toward tau_c as speed builds
    b       : viscous damping coefficient [N.m.s/rad]

At (near) zero velocity the kinetic curve is undefined in sign, so we apply a
stiction band: if |omega| < omega_eps and the net *applied* torque trying to
move the joint is below the break-away level tau_s, friction exactly cancels the
applied torque and the joint is held at rest (no motion, no chatter). Once the
applied torque exceeds tau_s, the joint breaks away and the kinetic curve
applies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrictionParameters:
    """Parameters for the Coulomb + viscous + Stribeck friction model."""

    coulomb_torque: float = 0.0       # tau_c [N.m]
    static_torque: float = 0.0        # tau_s [N.m], break-away level (>= tau_c)
    viscous_damping: float = 0.0      # b [N.m.s/rad]
    stribeck_velocity: float = 0.05   # omega_s [rad/s]
    velocity_eps: float = 1e-4        # omega_eps [rad/s], stiction band half-width


def kinetic_friction_torque(omega: float, p: FrictionParameters) -> float:
    """Coulomb + Stribeck + viscous friction for a *moving* joint (omega != 0).

    Does not include stiction handling -- use :func:`friction_torque` for the
    full model with correct zero-velocity behaviour.
    """
    stribeck = p.coulomb_torque + (p.static_torque - p.coulomb_torque) * np.exp(
        -((omega / p.stribeck_velocity) ** 2)
    )
    return stribeck * np.sign(omega) + p.viscous_damping * omega


def friction_torque(omega: float, applied_torque: float, p: FrictionParameters) -> float:
    """Full friction torque including correct zero-velocity (stiction) handling.

    Args:
        omega: Current angular velocity [rad/s].
        applied_torque: Net non-friction torque driving the joint at this instant
            (electromagnetic torque minus external load) [N.m]. Used only inside
            the stiction band to decide whether the joint breaks away.
        p: Friction parameters.

    Returns:
        Friction torque [N.m]. By convention this *opposes* motion, so the caller
        subtracts it: ``J * domega/dt = applied_torque - friction_torque(...)``.

    Stiction band behaviour (|omega| < velocity_eps):
        - If |applied_torque| <= tau_s (break-away level): the joint is stuck.
          Friction equals the applied torque (capped at tau_s) so the net torque
          is ~0 and the joint stays at rest. This prevents sign() chatter.
        - If |applied_torque| > tau_s: the joint breaks away; we return the static
          friction magnitude opposing the applied torque, so net torque is
          ``applied_torque - tau_s * sign(applied_torque)`` and motion begins.
    """
    if p.static_torque > 0.0 and abs(omega) < p.velocity_eps:
        # Inside stiction band: friction can hold the joint static up to tau_s.
        # (Only meaningful when there is static friction to hold it; a motor with
        # tau_s == 0 has nothing to stick and accelerates from any net torque.)
        if abs(applied_torque) <= p.static_torque:
            # Held static -- friction exactly cancels the applied torque.
            return applied_torque
        # Break-away: static friction opposes the applied torque at its max.
        return p.static_torque * np.sign(applied_torque)

    # Moving: standard kinetic Coulomb + Stribeck + viscous curve.
    return kinetic_friction_torque(omega, p)


def is_stuck(omega: float, applied_torque: float, p: FrictionParameters) -> bool:
    """Return True if the joint is held static by stiction at this instant.

    A joint can only be *held* static if there is static friction to hold it, so
    ``tau_s == 0`` is never stuck (it accelerates from any net torque).
    """
    return (
        p.static_torque > 0.0
        and abs(omega) < p.velocity_eps
        and abs(applied_torque) <= p.static_torque
    )
