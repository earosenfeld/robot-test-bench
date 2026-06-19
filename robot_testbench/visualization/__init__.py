"""
RobotTestBench visualization package.

The dashboard depends on the optional ``dash``/``plotly`` stack. To keep the core
package importable (and the test suite runnable) without those extras installed,
the dashboard symbols are imported lazily via ``__getattr__``.
"""

from .plots import TimeSeriesPlot, XYPlot, ScatterPlot

__all__ = [
    'launch_dashboard',
    'TimeSeriesPlot',
    'XYPlot',
    'ScatterPlot',
]


def __getattr__(name):
    if name == 'launch_dashboard':
        from .dashboard import launch_dashboard
        return launch_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
