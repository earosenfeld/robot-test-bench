#!/usr/bin/env python3
"""
RobotTestBench command-line entry point.

Exposed as the ``robot-testbench`` console script and via
``python -m robot_testbench``. By default it runs a short, headless brushed-DC
motor simulation and prints the steady-state result so the install can be
verified without launching the (optional) Dash dashboard.
"""

from __future__ import annotations

import argparse

from robot_testbench.motor import MotorParameters, MotorSimulator


def _demo_simulation(voltage: float = 12.0, duration: float = 2.0, dt: float = 1e-4) -> dict:
    """Run a short open-loop step and return the final motor state."""
    params = MotorParameters(
        inertia=0.01,
        damping=0.05,
        torque_constant=0.1,
        max_torque=10.0,
        max_speed=500.0,
        resistance=1.0,
        inductance=0.01,
    )
    motor = MotorSimulator(params)
    steps = int(round(duration / dt))
    for _ in range(steps):
        motor.step(dt, voltage)
    return motor.get_state()


def main(argv: list[str] | None = None) -> int:
    """Console entry point."""
    parser = argparse.ArgumentParser(
        prog="robot-testbench",
        description="RobotTestBench - actuator & sensor simulation test bench",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo", "dashboard"],
        help="demo: run a headless motor-step demo (default). "
        "dashboard: launch the Dash UI (requires the optional 'dash' extra).",
    )
    parser.add_argument("--voltage", type=float, default=12.0, help="Step voltage [V] for the demo.")
    parser.add_argument("--duration", type=float, default=2.0, help="Demo duration [s].")
    args = parser.parse_args(argv)

    if args.command == "dashboard":
        try:
            from robot_testbench.visualization.dashboard import launch_dashboard
        except ImportError as exc:  # pragma: no cover - optional dependency path
            print(f"Dashboard unavailable (install the 'dash' extra): {exc}")
            return 1
        launch_dashboard()
        return 0

    state = _demo_simulation(voltage=args.voltage, duration=args.duration)
    print("RobotTestBench demo - brushed-DC motor step response")
    print(f"  applied voltage : {args.voltage:.3f} V")
    print(f"  final velocity  : {state['velocity']:.3f} rad/s")
    print(f"  final current   : {state['current']:.3f} A")
    print(f"  output torque   : {state['torque']:.4f} N.m")
    print(f"  temperature     : {state['temperature']:.2f} degC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
