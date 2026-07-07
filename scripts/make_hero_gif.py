"""Generate assets/input_shaping_demo.gif.

Animated companion to the static resonance figure: the same rest-to-rest load
move on the two-mass flexible joint, unshaped vs ZV input-shaped, rendered as
two axis dials (needle = load angle) above the progressively drawn time
traces. Everything is computed by the real robot_testbench API.

Run from the repo root:

    python scripts/make_hero_gif.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from robot_testbench.resonance import (
    FlexibleJointParameters,
    apply_input_shaper,
    residual_vibration,
    resonance_modes,
    simulate_load_response,
)

OUT = Path(__file__).resolve().parents[1] / "assets" / "input_shaping_demo.gif"

BG = "#0d1117"
FG = "#e6edf3"
C_UNSHAPED = "#f85149"
C_SHAPED = "#3fb950"


def main() -> None:
    p = FlexibleJointParameters(
        motor_inertia=0.05, load_inertia=0.02, stiffness=400.0, damping=0.008,
    )
    m = resonance_modes(p)

    dt = 1e-4
    T = 0.9
    n = int(T / dt)
    t = np.arange(n) * dt
    cmd = np.zeros(n)
    # Half the resonant period per bang: maximal excitation of the mode.
    ta = int(round(np.pi / m.omega_res / dt))
    cmd[:ta] = 1.0
    cmd[ta : 2 * ta] = -1.0

    _, th_u = simulate_load_response(p, cmd, dt)
    shaped = apply_input_shaper(cmd, dt, m.omega_res, m.zeta, kind="ZV")
    _, th_s = simulate_load_response(p, shaped, dt)

    rv_u = residual_vibration(th_u, 0.3)
    rv_s = residual_vibration(th_s, 0.3)
    final = float(th_u[int(0.8 * n):].mean())

    # Dial needle angle: map load angle so the move sweeps ~100 degrees.
    scale = np.deg2rad(100) / final

    fig = plt.figure(figsize=(8.0, 5.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35)
    ax_du = fig.add_subplot(gs[0, 0], projection="polar")
    ax_ds = fig.add_subplot(gs[0, 1], projection="polar")
    ax_tr = fig.add_subplot(gs[1, :])

    needles = []
    for ax, color, title in (
        (ax_du, C_UNSHAPED, "Unshaped command"),
        (ax_ds, C_SHAPED, "ZV input-shaped"),
    ):
        ax.set_facecolor("#161b22")
        ax.set_theta_zero_location("W")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-15)
        ax.set_thetamax(115)
        ax.set_rticks([])
        ax.set_xticks([0.0, scale * final])
        ax.set_xticklabels(["start", "target"], color=FG, fontsize=8)
        ax.spines["polar"].set_color("#30363d")
        ax.set_title(title, color=color, fontsize=10, pad=10)
        ax.plot([scale * final] * 2, [0, 1], color="#8b949e", ls="--", lw=1.0)
        (needle,) = ax.plot([0, 0], [0, 0.95], color=color, lw=3.0,
                            solid_capstyle="round")
        needles.append(needle)

    ax_tr.set_facecolor("#161b22")
    for spine in ax_tr.spines.values():
        spine.set_color("#30363d")
    ax_tr.tick_params(colors=FG, labelsize=8)
    ax_tr.axhline(final, color="#8b949e", ls="--", lw=1.0)
    (ln_u,) = ax_tr.plot([], [], color=C_UNSHAPED, lw=1.6,
                         label=f"unshaped — residual {rv_u*1e3:.1f} mrad")
    (ln_s,) = ax_tr.plot([], [], color=C_SHAPED, lw=1.6,
                         label=f"ZV-shaped — residual {rv_s*1e3:.2f} mrad "
                               f"(−{100*(1-rv_s/rv_u):.1f}%)")
    ax_tr.set_xlim(0, T)
    ax_tr.set_ylim(min(th_u.min(), 0) * 1.1, th_u.max() * 1.15)
    ax_tr.set_xlabel("time [s]", color=FG, fontsize=9)
    ax_tr.set_ylabel("load angle [rad]", color=FG, fontsize=9)
    ax_tr.legend(loc="lower right", fontsize=8, facecolor="#161b22",
                 edgecolor="#30363d", labelcolor=FG)
    fig.suptitle(
        f"Same move, two commands — ZV shaper notched at "
        f"$\\omega_{{res}}$ = {m.omega_res:.0f} rad/s",
        color=FG, fontsize=11,
    )

    n_frames = 60

    def update(frame: int):
        k = max(2, int((frame + 1) / n_frames * n))
        ln_u.set_data(t[:k], th_u[:k])
        ln_s.set_data(t[:k], th_s[:k])
        for needle, th in zip(needles, (th_u, th_s)):
            a = scale * th[k - 1]
            needle.set_data([a, a], [0, 0.95])
        return [ln_u, ln_s, *needles]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    OUT.parent.mkdir(exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=15), dpi=80)
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
