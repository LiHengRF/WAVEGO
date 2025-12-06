"""
WaveGo Agent - Web Dashboard Module
"""

from .server import (
    DashboardManager,
    dashboard,
    app,
    create_dashboard_server,
    run_standalone
)

__all__ = [
    'DashboardManager',
    'dashboard', 
    'app',
    'create_dashboard_server',
    'run_standalone'
]
