"""
Generate publication-quality figures for the README from the REAL simulation API.

Every figure is produced by driving the actual `robot_testbench` models -- there
are no hand-authored data arrays. Run headless:

    .venv/bin/python scripts/make_figures.py

Outputs five PNGs into ``assets/``:
    motor_step_response.png      second-order step (velocity + winding current)
    friction_curve.png           Coulomb + Stribeck + viscous S-curve
    cascade_step.png             closed-loop position step, anti-windup vs windup
    system_id.png                least-squares parameter recovery vs ground truth
    resonance_suppression.png    two-mass FRF + ZV input-shaping of a load move
"""

from __future__ import annotations

import os
import sys

import numpy as np

# --- House style -----------------------------------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.frameon": False, "lines.linewidth": 2.0,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

# --- Make the package importable when run from anywhere ---------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from robot_testbench.motor.motor_simulator import MotorParameters, MotorSimulator
from robot_testbench.motor.friction import FrictionParameters, kinetic_friction_torque
from robot_testbench.control.cascade import CascadeController, CascadeGains
from robot_testbench.analysis.system_id import (
    generate_chirp_voltage,
    collect_excitation_data,
    identify_motor_parameters,
)
from robot_testbench.resonance import (
    FlexibleJointParameters,
    resonance_modes,
    compliance_ratio,
    simulate_load_response,
    apply_input_shaper,
    residual_vibration,
)

ASSETS = os.path.join(_REPO_ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(ASSETS, name)
    fig.savefig(path)
    plt.close(fig)
    kb = os.path.getsize(path) / 1024.0
    print(f"  wrote {name:28s} ({kb:6.1f} KB)")
    return path


# ---------------------------------------------------------------------------
# 1. Motor step response: second-order rise in velocity, winding current spike.
# ---------------------------------------------------------------------------
def fig_motor_step() -> None:
    p = MotorParameters(
        inertia=0.05, damping=0.2, torque_constant=0.1, max_torque=40.0,
        max_speed=500.0, resistance=1.0, inductance=0.01,
    )
    motor = MotorSimulator(p)
    dt, V = 5e-4, 6.0
    n = 2000  # 1.0 s
    t = np.arange(n) * dt
    vel = np.empty(n)
    cur = np.empty(n)
    for k in range(n):
        _, vel[k], cur[k] = motor.step(dt, V)

    # Closed-form steady-state velocity for reference (Ke == Kt, viscous only).
    denom = p.resistance * p.damping + p.torque_constant * p.ke
    omega_ss = V * p.torque_constant / denom  # motor-side == output-side (N=1)

    fig, ax1 = plt.subplots(figsize=(7.2, 4.3))
    l1, = ax1.plot(t, vel, color=PALETTE[0], label="Output velocity")
    ax1.axhline(omega_ss, color=PALETTE[0], ls="--", lw=1.2, alpha=0.6,
                label="Steady-state $\\omega_{ss}$")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Velocity [rad/s]", color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])
    ax1.set_xlim(0, t[-1])

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    l2, = ax2.plot(t, cur, color=PALETTE[1], label="Winding current")
    ax2.set_ylabel("Current [A]", color=PALETTE[1])
    ax2.tick_params(axis="y", labelcolor=PALETTE[1])

    ax1.set_title("Brushed-DC motor step response (6 V, RK4 plant)")
    lines = [l1, l2, ax1.get_lines()[1]]
    ax1.legend(handles=lines, loc="center right")
    _save(fig, "motor_step_response.png")


# ---------------------------------------------------------------------------
# 2. Friction curve: static break-away -> Stribeck dip -> viscous rise.
# ---------------------------------------------------------------------------
def fig_friction_curve() -> None:
    fp = FrictionParameters(
        coulomb_torque=0.5, static_torque=1.0,
        viscous_damping=0.1, stribeck_velocity=0.6,
    )
    omega = np.linspace(-6.0, 6.0, 1201)
    # kinetic_friction_torque is scalar-by-design; evaluate elementwise.
    tau = np.array([kinetic_friction_torque(float(w), fp) for w in omega])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(omega, tau, color=PALETTE[4], label="Kinetic friction $\\tau_f(\\omega)$")
    ax.axhline(0.0, color="#94a3b8", lw=0.8)
    ax.axvline(0.0, color="#94a3b8", lw=0.8)

    # Reference levels (positive branch).
    ax.axhline(fp.static_torque, color=PALETTE[1], ls=":", lw=1.3, alpha=0.8)
    ax.axhline(fp.coulomb_torque, color=PALETTE[2], ls=":", lw=1.3, alpha=0.8)

    # Annotate the three regimes on the positive-omega branch. Text is kept
    # inside the axes (away from the title) with arrows pointing to the curve.
    ax.annotate(
        "Static break-away $\\tau_s$",
        xy=(0.04, fp.static_torque), xytext=(-5.6, 1.18),
        arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.2),
        color=PALETTE[1], fontsize=10,
    )
    # Stribeck dip toward Coulomb: minimum of the magnitude on the + branch.
    pos = omega > 1e-6
    dip_idx = np.argmin(tau[pos])
    w_dip = omega[pos][dip_idx]
    ax.annotate(
        "Stribeck dip $\\to \\tau_c$",
        xy=(w_dip, tau[pos][dip_idx]),
        xytext=(0.5, -0.7),
        arrowprops=dict(arrowstyle="->", color=PALETTE[2], lw=1.2),
        color=PALETTE[2], fontsize=10,
    )
    ax.annotate(
        "Viscous rise $b\\,\\omega$",
        xy=(5.2, tau[np.argmin(np.abs(omega - 5.2))]),
        xytext=(3.4, 0.18),
        arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=1.2),
        color=PALETTE[0], fontsize=10,
    )

    ax.set_xlabel("Angular velocity $\\omega$ [rad/s]")
    ax.set_ylabel("Friction torque $\\tau_f$ [N$\\cdot$m]")
    ax.set_title("Coulomb + Stribeck + viscous friction model")
    ax.set_xlim(omega[0], omega[-1])
    ax.set_ylim(-1.5, 1.4)
    ax.legend(loc="lower right")
    _save(fig, "friction_curve.png")


# ---------------------------------------------------------------------------
# 3. Cascade position step: back-calculation anti-windup vs disabled.
# ---------------------------------------------------------------------------
def _run_cascade(tracking_time_constant: float, setpoint: float = 1.0):
    p = MotorParameters(
        inertia=0.05, damping=0.2, torque_constant=0.1, max_torque=4.0,
        max_speed=50.0, resistance=1.0, inductance=0.001,
    )
    gains = CascadeGains(
        pos_kp=25.0, pos_ki=0.0,
        vel_kp=0.6, vel_ki=6.0,
        cur_kp=6.0, cur_ki=1200.0,
    )
    ctrl = CascadeController(
        gains, kt=p.torque_constant, ke=p.ke, inertia=p.inertia,
        max_velocity=50.0, max_torque=4.0, max_voltage=24.0,
        tracking_time_constant=tracking_time_constant,
    )
    ctrl.set_target(setpoint)
    motor = MotorSimulator(p)
    dt, n = 5e-4, 5000  # 2.5 s
    t = np.arange(n) * dt
    pos_log = np.empty(n)
    pos = vel = cur = omega_m = 0.0
    for k in range(n):
        voltage = ctrl.update(pos, vel, cur, omega_m, dt)
        pos, vel, cur = motor.step(dt, voltage)
        omega_m = motor.get_state()["motor_velocity"]
        pos_log[k] = pos
    return t, pos_log


def fig_cascade_step() -> None:
    setpoint = 1.0
    # Correct anti-windup: small tracking time constant bleeds the integrator.
    t, pos_aw = _run_cascade(tracking_time_constant=0.05, setpoint=setpoint)
    # Windup case: huge Tt effectively disables back-calculation.
    _, pos_wu = _run_cascade(tracking_time_constant=1e9, setpoint=setpoint)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.axhline(setpoint, color="#475569", ls="--", lw=1.3, label=f"Setpoint = {setpoint:g} rad")
    ax.plot(t, pos_wu, color=PALETTE[1], label="Integrator windup ($T_t\\to\\infty$)")
    ax.plot(t, pos_aw, color=PALETTE[0], label="Back-calculation anti-windup")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output position [rad]")
    ax.set_title("Cascade position step: anti-windup vs windup")
    ax.set_xlim(0, t[-1])
    ax.legend(loc="lower right")
    _save(fig, "cascade_step.png")


# ---------------------------------------------------------------------------
# 4. System identification: recovered vs true parameters (normalized bars).
# ---------------------------------------------------------------------------
def fig_system_id() -> None:
    true = MotorParameters(
        inertia=0.04, damping=0.25, torque_constant=0.15, max_torque=40.0,
        max_speed=500.0, resistance=1.2, inductance=0.02,
    )
    motor = MotorSimulator(true)
    dt = 1e-4
    volt = generate_chirp_voltage(duration=4.0, dt=dt, amplitude=6.0, offset=5.0)
    v_log, i_log, w_log = collect_excitation_data(motor, volt, dt)
    est = identify_motor_parameters(v_log, i_log, w_log, dt)

    labels = ["R\n[$\\Omega$]", "L\n[H]", "Kt\n[N$\\cdot$m/A]", "J\n[kg$\\cdot$m$^2$]", "b\n[N$\\cdot$m$\\cdot$s]"]
    true_vals = np.array([true.resistance, true.inductance, true.torque_constant,
                          true.inertia, true.damping])
    est_vals = np.array([est.resistance, est.inductance, est.torque_constant,
                         est.inertia, est.damping])
    err_pct = 100.0 * (est_vals - true_vals) / true_vals

    # Normalize each parameter to its true value so all bars are visible together.
    true_norm = np.ones_like(true_vals)
    est_norm = est_vals / true_vals

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - w / 2, true_norm, w, color=PALETTE[2], label="True")
    ax.bar(x + w / 2, est_norm, w, color=PALETTE[0], label="Identified")
    ax.axhline(1.0, color="#94a3b8", lw=0.8)

    for xi, (tv, e) in enumerate(zip(true_vals, err_pct)):
        ax.annotate(f"{e:+.2f}%", xy=(xi, 1.02), ha="center", va="bottom",
                    fontsize=9, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value normalized to truth")
    ax.set_ylim(0, 1.18)
    ax.set_title("Least-squares system ID: recovered vs true (chirp excitation)")
    ax.legend(loc="lower right", ncol=2)
    _save(fig, "system_id.png")


# ---------------------------------------------------------------------------
# 5. Flexible-joint resonance: FRF peak + anti-resonance, and input shaping.
# ---------------------------------------------------------------------------
def fig_resonance_suppression() -> None:
    # Two-mass plant: a compliant shaft couples a 0.05 motor inertia to a 0.02
    # load through a 400 N.m/rad spring with light structural damping.
    p = FlexibleJointParameters(
        motor_inertia=0.05, load_inertia=0.02, stiffness=400.0, damping=0.05,
    )
    m = resonance_modes(p)

    # --- Left panel: dynamic-compliance FRF (normalised by the rigid baseline).
    omega = np.linspace(1.0, 2.6 * m.omega_res, 60000)
    load_mag = compliance_ratio(p, omega, output="load")    # resonance peak
    motor_mag = compliance_ratio(p, omega, output="motor")  # anti-resonance dip

    # --- Right panel: rest-to-rest move, unshaped (ringing) vs ZV-shaped (clean).
    dt = 1e-4
    T = 1.2
    n = int(T / dt)
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    ta = int(0.05 / dt)
    cmd[:ta] = 1.0
    cmd[ta : 2 * ta] = -1.0  # net-zero torque impulse -> ends at rest at new angle

    _, theta_l_u = simulate_load_response(p, cmd, dt)
    shaped = apply_input_shaper(cmd, dt, m.omega_res, m.zeta, kind="ZV")
    _, theta_l_s = simulate_load_response(p, shaped, dt)

    rv_u = residual_vibration(theta_l_u, 0.4)
    rv_s = residual_vibration(theta_l_s, 0.4)
    reduction = 100.0 * (1.0 - rv_s / rv_u)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 4.6))
    fig.subplots_adjust(wspace=0.26)

    # Left: FRF, log-y to show the sharp peak and deep notch together.
    axL.semilogy(omega, load_mag, color=PALETTE[0],
                 label="Load FRF $\\tau\\!\\to\\!\\theta_l$ (resonance)")
    axL.semilogy(omega, motor_mag, color=PALETTE[2],
                 label="Motor FRF $\\tau\\!\\to\\!\\theta_m$ (anti-resonance)")
    axL.axvline(m.omega_res, color=PALETTE[1], ls="--", lw=1.3, alpha=0.8)
    axL.axvline(m.omega_antires, color=PALETTE[4], ls=":", lw=1.3, alpha=0.8)
    axL.annotate(f"$\\omega_{{res}}$ = {m.omega_res:.0f} rad/s",
                 xy=(m.omega_res, load_mag.max()),
                 xytext=(m.omega_res * 1.25, load_mag.max() * 0.28),
                 color=PALETTE[1], fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=PALETTE[1], lw=1.2))
    dip_idx = np.argmin(motor_mag[(omega > 0.6 * m.omega_antires)
                                  & (omega < 1.4 * m.omega_antires)])
    axL.annotate(f"$\\omega_{{anti}}$ = {m.omega_antires:.0f} rad/s",
                 xy=(m.omega_antires, motor_mag[(omega > 0.6 * m.omega_antires)
                     & (omega < 1.4 * m.omega_antires)][dip_idx]),
                 xytext=(m.omega_antires * 0.30, 0.18),
                 color=PALETTE[4], fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=1.2))
    axL.set_xlabel("Frequency $\\omega$ [rad/s]")
    axL.set_ylabel("Compliance amplification $|H|/|H_{rigid}|$")
    axL.set_title("Two-mass FRF: peak + anti-resonance")
    axL.set_xlim(omega[0], omega[-1])
    axL.legend(loc="upper right", fontsize=9)

    # Right: load step response, unshaped vs input-shaped.
    final = theta_l_u[int(0.8 * n):].mean()
    axR.axhline(final, color="#475569", ls="--", lw=1.2,
                label=f"Target = {final:.3f} rad")
    axR.plot(t, theta_l_u, color=PALETTE[1],
             label=f"Unshaped (residual {rv_u*1e3:.2f} mrad)")
    axR.plot(t, theta_l_s, color=PALETTE[0],
             label=f"ZV input-shaped (residual {rv_s*1e3:.3f} mrad)")
    axR.set_xlabel("Time [s]")
    axR.set_ylabel("Load angle $\\theta_l$ [rad]")
    axR.set_title("Load move: input shaping kills the ringing")
    axR.set_xlim(0, t[-1])
    axR.annotate(f"residual vibration\n$-${reduction:.1f}%",
                 xy=(0.62, 0.30), xycoords="axes fraction",
                 ha="center", fontsize=11, color=PALETTE[2], fontweight="bold")
    axR.legend(loc="lower right")

    _save(fig, "resonance_suppression.png")


def main() -> None:
    print("Generating README figures from the real robot_testbench API:")
    fig_motor_step()
    fig_friction_curve()
    fig_cascade_step()
    fig_system_id()
    fig_resonance_suppression()
    print("Done.")


if __name__ == "__main__":
    main()
