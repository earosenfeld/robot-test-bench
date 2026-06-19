"""Allow ``python -m robot_testbench`` to invoke the CLI entry point."""

from robot_testbench.main import main

if __name__ == "__main__":
    raise SystemExit(main())
