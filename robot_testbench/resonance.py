"""
Two-mass flexible-joint model with resonance analysis and input shaping.

A real servo axis is never a single rigid inertia: the motor rotor and the
driven load are coupled through a *compliant* transmission (belt, harmonic
drive, long shaft). That compliance turns the plant into a **two-mass
resonant system** -- the classic source of audible "singing" axes, limit-cycle
chatter, and the hard ceiling on closed-loop bandwidth. This module models that
plant, exposes its resonance/anti-resonance analytically, computes its frequency
response, and demonstrates **input shaping** (ZV / ZVD) as a feed-forward way to
command a move that does not excite the resonance.

Physics (SI units), with motor angle theta_m and load angle theta_l:

    Jm * theta_m'' = tau - k (theta_m - theta_l) - c (theta_m' - theta_l')
    Jl * theta_l'' =       k (theta_m - theta_l) + c (theta_m' - theta_l')

  - Jm, Jl   : motor-side and load-side inertias [kg.m^2]
  - k        : shaft torsional stiffness [N.m/rad]
  - c        : shaft internal (structural) damping [N.m.s/rad]
  - tau      : torque applied at the motor shaft [N.m]

State vector layout (used for the RK4 integrator):

    x = [theta_m, omega_m, theta_l, omega_l]

Two transfer functions matter, and they are *different* -- this is the whole
point of a non-collocated drive:

  * Collocated (motor side), tau -> theta_m, carries the **anti-resonance** zero
    at omega_antires = sqrt(k / Jl): at that frequency the load acts as a tuned
    absorber and the motor barely moves. Numerator ~ (Jl s^2 + c s + k).

  * Non-collocated (load side), tau -> theta_l, is what we actually want to move.
    Numerator ~ (c s + k) -- no anti-resonance -- and the **resonance** pole at
    omega_res = sqrt(k (Jm + Jl) / (Jm Jl)) shows up as a sharp peak.

Both share the denominator  s^2 [ Jm Jl s^2 + c (Jm+Jl) s + k (Jm+Jl) ]  (two
rigid-body poles at the origin plus the resonant pair). The damped resonant pair
has natural frequency omega_res and damping ratio

    zeta = c (Jm + Jl) / (2 * sqrt(k (Jm + Jl) Jm Jl)).

Input shaping (ZV/ZVD) convolves the reference command with a short impulse
train whose impulses are timed and weighted so their individual residual
vibrations cancel at the design frequency -- a zero-phase-error, model-based
notch placed exactly on the resonance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motor.integrators import rk4_step

# State indices for [theta_m, omega_m, theta_l, omega_l].
_TM, _WM, _TL, _WL = 0, 1, 2, 3


# ===========================================================================
# Plant model
# ===========================================================================
@dataclass
class FlexibleJointParameters:
    """Two-mass flexible-joint (compliant transmission) parameters."""

    motor_inertia: float       # Jm [kg.m^2]
    load_inertia: float        # Jl [kg.m^2]
    stiffness: float           # k  [N.m/rad]   shaft torsional stiffness
    damping: float = 0.0       # c  [N.m.s/rad] shaft internal damping
    motor_damping: float = 0.0  # bm [N.m.s/rad] viscous drag to ground on rotor
    load_damping: float = 0.0   # bl [N.m.s/rad] viscous drag to ground on load


@dataclass
class ResonanceModes:
    """Analytic modal description of the two-mass plant."""

    omega_res: float        # undamped resonant natural frequency [rad/s]
    omega_antires: float    # anti-resonance frequency (motor-side zero) [rad/s]
    zeta: float             # damping ratio of the resonant pair [-]
    omega_damped: float     # damped resonant frequency omega_res*sqrt(1-zeta^2)
    f_res: float            # omega_res / (2 pi) [Hz]
    f_antires: float        # omega_antires / (2 pi) [Hz]


def resonance_modes(params: FlexibleJointParameters) -> ResonanceModes:
    """Return the analytic resonance / anti-resonance of the two-mass plant.

    omega_res     = sqrt( k (Jm + Jl) / (Jm Jl) )      (resonant pole)
    omega_antires = sqrt( k / Jl )                      (motor-side zero)
    zeta          = c (Jm + Jl) / (2 sqrt(k (Jm+Jl) Jm Jl))

    The internal shaft damping ``c`` is what damps the resonant pair; the
    ground-referenced viscous terms (motor_damping/load_damping) shift the poles
    only slightly and are ignored in this closed-form modal summary (they are of
    course present in the time-domain integrator).
    """
    Jm, Jl, k, c = (
        params.motor_inertia,
        params.load_inertia,
        params.stiffness,
        params.damping,
    )
    if Jm <= 0.0 or Jl <= 0.0 or k <= 0.0:
        raise ValueError("inertias and stiffness must be positive")

    omega_res = np.sqrt(k * (Jm + Jl) / (Jm * Jl))
    omega_antires = np.sqrt(k / Jl)
    # zeta of the resonant pair: c*(Jm+Jl) / (2 * sqrt(k*(Jm+Jl)*Jm*Jl)).
    zeta = c * (Jm + Jl) / (2.0 * np.sqrt(k * (Jm + Jl) * Jm * Jl))
    zeta = float(zeta)
    omega_damped = omega_res * np.sqrt(max(0.0, 1.0 - zeta ** 2))
    return ResonanceModes(
        omega_res=float(omega_res),
        omega_antires=float(omega_antires),
        zeta=zeta,
        omega_damped=float(omega_damped),
        f_res=float(omega_res / (2.0 * np.pi)),
        f_antires=float(omega_antires / (2.0 * np.pi)),
    )


class FlexibleJointSimulator:
    """Time-domain two-mass flexible-joint plant integrated with fixed-step RK4.

    The input is the motor-shaft torque ``tau`` (held constant over each step,
    zero-order hold). State and the convenience getters are exact for the linear
    plant up to the O(dt^4) RK4 truncation error.
    """

    def __init__(self, params: FlexibleJointParameters):
        self.params = params
        self.state = np.zeros(4)  # [theta_m, omega_m, theta_l, omega_l]
        self._sim_time = 0.0

    # ----------------------------------------------------------- derivative
    def _state_derivative(self, tau: float):
        p = self.params
        Jm, Jl, k, c = p.motor_inertia, p.load_inertia, p.stiffness, p.damping
        bm, bl = p.motor_damping, p.load_damping

        def f(x: np.ndarray) -> np.ndarray:
            theta_m, omega_m, theta_l, omega_l = x[_TM], x[_WM], x[_TL], x[_WL]
            dtheta = theta_m - theta_l
            domega = omega_m - omega_l
            tau_shaft = k * dtheta + c * domega  # torque the shaft exerts on load
            domega_m = (tau - tau_shaft - bm * omega_m) / Jm
            domega_l = (tau_shaft - bl * omega_l) / Jl
            return np.array([omega_m, domega_m, omega_l, domega_l])

        return f

    # ----------------------------------------------------------------- step
    def step(self, dt: float, tau: float) -> tuple[float, float]:
        """Advance one RK4 step under motor torque ``tau``.

        Returns ``(theta_m, theta_l)`` -- motor and load angle after the step.
        """
        f = self._state_derivative(float(tau))
        self.state = rk4_step(f, self.state, dt)
        self._sim_time += dt
        return float(self.state[_TM]), float(self.state[_TL])

    def reset(self) -> None:
        self.state = np.zeros(4)
        self._sim_time = 0.0

    def get_state(self) -> dict:
        return {
            "motor_angle": float(self.state[_TM]),
            "motor_velocity": float(self.state[_WM]),
            "load_angle": float(self.state[_TL]),
            "load_velocity": float(self.state[_WL]),
            "time": self._sim_time,
        }


def simulate_load_response(
    params: FlexibleJointParameters,
    torque: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive the plant with a ZOH torque sequence; log motor & load angle.

    Args:
        params: Plant parameters.
        torque: Per-step motor torque [N.m], length ``n``.
        dt: Time step [s].

    Returns:
        ``(theta_m, theta_l)`` angle trajectories, each length ``n`` (sampled at
        the end of each step).
    """
    sim = FlexibleJointSimulator(params)
    torque = np.asarray(torque, dtype=float)
    n = torque.size
    theta_m = np.empty(n)
    theta_l = np.empty(n)
    for i in range(n):
        tm, tl = sim.step(dt, float(torque[i]))
        theta_m[i] = tm
        theta_l[i] = tl
    return theta_m, theta_l


# ===========================================================================
# Frequency response (analytic transfer functions)
# ===========================================================================
def _denominator(params: FlexibleJointParameters) -> np.ndarray:
    """Resonant-quadratic of the two-mass plant (descending powers of s), i.e.
    the denominator WITHOUT the rigid-body s^2 factor common to both TFs.

        D(s) = Jm Jl s^2 + c (Jm + Jl) s + k (Jm + Jl)

    This is the exact resonant quadratic for the structurally-damped plant
    (shaft damping ``c`` only). The optional ground-referenced viscous terms
    (motor_damping/load_damping) are modelled exactly by the time-domain
    integrator but are intentionally NOT folded into this closed-form quadratic:
    they break the clean two-mass factorisation (the plant is no longer a pure
    double integrator at DC) and are off by default, so the analytic FRF and
    modal formulas describe the ``bm = bl = 0`` case the tests exercise.
    """
    Jm, Jl, k, c = (
        params.motor_inertia, params.load_inertia, params.stiffness, params.damping,
    )
    a2 = Jm * Jl
    a1 = c * (Jm + Jl)
    a0 = k * (Jm + Jl)
    return np.array([a2, a1, a0])


def transfer_function(
    params: FlexibleJointParameters,
    output: str = "load",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(num, den)`` polynomials (descending powers of s) for tau->angle.

    ``output="load"``  -> tau -> theta_l (non-collocated): num = (c s + k),
        resonance peak, no anti-resonance.
    ``output="motor"`` -> tau -> theta_m (collocated): num = (Jl s^2 + c s + k),
        carries the anti-resonance zero at sqrt(k/Jl).

    The full denominator includes the s^2 rigid-body factor:
        den = s^2 * (Jm Jl s^2 + c(Jm+Jl) s + k(Jm+Jl)).
    """
    Jl, k, c = params.load_inertia, params.stiffness, params.damping
    quad = _denominator(params)              # [a2, a1, a0]
    den = np.polymul(quad, [1.0, 0.0, 0.0])  # multiply by s^2

    if output == "load":
        num = np.array([c, k])               # c s + k
    elif output == "motor":
        num = np.array([Jl, c, k])           # Jl s^2 + c s + k
    else:
        raise ValueError("output must be 'load' or 'motor'")
    return num, den


def frequency_response(
    params: FlexibleJointParameters,
    omega: np.ndarray,
    output: str = "load",
) -> np.ndarray:
    """Complex frequency response H(j*omega) of tau->angle at the given omegas.

    Evaluated directly from the analytic transfer function (no FFT needed), so
    it is exact. ``omega`` is in rad/s.
    """
    num, den = transfer_function(params, output=output)
    jw = 1j * np.asarray(omega, dtype=float)
    return np.polyval(num, jw) / np.polyval(den, jw)


def frequency_response_magnitude(
    params: FlexibleJointParameters,
    omega: np.ndarray,
    output: str = "load",
) -> np.ndarray:
    """Magnitude |H(j*omega)| of the tau->angle frequency response."""
    return np.abs(frequency_response(params, omega, output=output))


def rigid_body_response(params: FlexibleJointParameters, omega: np.ndarray) -> np.ndarray:
    """Complex tau->angle response if the joint were perfectly rigid.

    With an infinitely stiff shaft the two masses move as one lumped inertia
    (Jm + Jl), giving the pure double-integrator  H_rigid(s) = 1 / [(Jm+Jl) s^2].
    Dividing the flexible response by this baseline isolates the *resonant*
    behaviour from the rigid-body 1/s^2 roll-off.
    """
    J = params.motor_inertia + params.load_inertia
    jw = 1j * np.asarray(omega, dtype=float)
    return 1.0 / (J * jw ** 2)


def compliance_ratio(
    params: FlexibleJointParameters,
    omega: np.ndarray,
    output: str = "load",
) -> np.ndarray:
    """Flexible-joint FRF normalised by the rigid-body baseline (dimensionless).

    ``|H_flex(j omega)| / |H_rigid(j omega)|``. This is the dynamic-compliance
    amplification servo engineers actually plot: it is ~1 (0 dB) at low
    frequency, rises to a sharp **peak at omega_res**, and -- for the collocated
    (``output="motor"``) response -- drops into a deep **anti-resonance notch at
    omega_antires**. Removing the 1/s^2 rigid roll-off is what makes the
    resonance the global maximum instead of being buried under the DC gain.
    """
    flex = frequency_response(params, omega, output=output)
    rigid = rigid_body_response(params, omega)
    return np.abs(flex) / np.abs(rigid)


# ===========================================================================
# Input shaping (ZV / ZVD)
# ===========================================================================
@dataclass
class InputShaper:
    """An impulse-train input shaper: impulse times [s] and amplitudes [-]."""

    times: np.ndarray       # impulse times, increasing, times[0] == 0
    amplitudes: np.ndarray  # impulse amplitudes, sum to 1

    @property
    def duration(self) -> float:
        """Total shaper length (time of the last impulse) [s]."""
        return float(self.times[-1])


def _damped_period(omega: float, zeta: float) -> float:
    """Damped period T_d = 2 pi / omega_d, omega_d = omega*sqrt(1-zeta^2)."""
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    zeta = float(np.clip(zeta, 0.0, 0.999999))
    omega_d = omega * np.sqrt(1.0 - zeta ** 2)
    return 2.0 * np.pi / omega_d


def zv_shaper(omega: float, zeta: float = 0.0) -> InputShaper:
    """Zero-Vibration (ZV) shaper: two impulses.

        K  = exp(-zeta pi / sqrt(1 - zeta^2))
        T  = T_d / 2 = pi / omega_d
        A1 = 1 / (1 + K)        at t = 0
        A2 = K / (1 + K)        at t = T

    Amplitudes are normalised to sum to 1 (unity DC gain -> no steady-state
    offset). Cancels the first-order residual vibration at ``omega``/``zeta``.
    """
    zeta = float(np.clip(zeta, 0.0, 0.999999))
    Td = _damped_period(omega, zeta)
    K = np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta ** 2))
    denom = 1.0 + K
    times = np.array([0.0, Td / 2.0])
    amps = np.array([1.0 / denom, K / denom])
    return InputShaper(times=times, amplitudes=amps)


def zvd_shaper(omega: float, zeta: float = 0.0) -> InputShaper:
    """Zero-Vibration-and-Derivative (ZVD) shaper: three impulses.

        A1 = 1 / D,  A2 = 2K / D,  A3 = K^2 / D,   D = 1 + 2K + K^2
        impulse times: 0, T_d/2, T_d

    ZVD additionally zeroes the *derivative* of the residual vibration w.r.t.
    frequency, so it is far more robust to a mis-estimated resonance than ZV, at
    the cost of being one half-period longer.
    """
    zeta = float(np.clip(zeta, 0.0, 0.999999))
    Td = _damped_period(omega, zeta)
    K = np.exp(-zeta * np.pi / np.sqrt(1.0 - zeta ** 2))
    D = 1.0 + 2.0 * K + K ** 2
    times = np.array([0.0, Td / 2.0, Td])
    amps = np.array([1.0 / D, 2.0 * K / D, (K ** 2) / D])
    return InputShaper(times=times, amplitudes=amps)


def apply_input_shaper(
    command: np.ndarray,
    dt: float,
    omega: float,
    zeta: float = 0.0,
    kind: str = "ZV",
) -> np.ndarray:
    """Convolve a sampled command with a ZV/ZVD input shaper.

    The continuous impulse train is sampled onto the ``dt`` grid (each impulse
    lands on the nearest sample, its amplitude renormalised so the discrete
    weights still sum to 1) and convolved with ``command``. The returned signal
    is the same length as ``command`` (the move is delayed by the shaper
    duration but the steady-state target is preserved exactly).

    Args:
        command: Reference command (e.g. a position step or torque pulse).
        dt: Sample period [s].
        omega: Design resonant frequency [rad/s].
        zeta: Design damping ratio [-].
        kind: ``"ZV"`` (two impulses) or ``"ZVD"`` (three impulses).

    Returns:
        The shaped command, same length as ``command``.
    """
    shaper = build_shaper(omega, zeta, kind)
    command = np.asarray(command, dtype=float)

    # Map each impulse to its nearest sample delay, accumulate weights there.
    delays = np.round(shaper.times / dt).astype(int)
    kernel_len = int(delays.max()) + 1
    kernel = np.zeros(kernel_len)
    for d, a in zip(delays, shaper.amplitudes):
        kernel[d] += a
    kernel /= kernel.sum()  # renormalise after grid quantisation -> unity DC gain

    shaped_full = np.convolve(command, kernel)
    return shaped_full[: command.size]


def build_shaper(omega: float, zeta: float = 0.0, kind: str = "ZV") -> InputShaper:
    """Construct a shaper by name (``"ZV"`` or ``"ZVD"``)."""
    k = kind.upper()
    if k == "ZV":
        return zv_shaper(omega, zeta)
    if k == "ZVD":
        return zvd_shaper(omega, zeta)
    raise ValueError("kind must be 'ZV' or 'ZVD'")


# ===========================================================================
# Residual-vibration metric
# ===========================================================================
def residual_vibration(
    signal: np.ndarray,
    settle_fraction: float = 0.5,
) -> float:
    """Peak-to-peak residual oscillation of ``signal`` over its settling tail.

    Looks only at the last ``settle_fraction`` of the record (where a
    well-damped or well-shaped move should be flat) and returns the
    peak-to-peak excursion about the tail's mean -- a direct measure of leftover
    ringing. A clean move -> ~0; a ringing one -> large.
    """
    sig = np.asarray(signal, dtype=float)
    n = sig.size
    start = int(n * (1.0 - settle_fraction))
    tail = sig[start:]
    return float(tail.max() - tail.min())
