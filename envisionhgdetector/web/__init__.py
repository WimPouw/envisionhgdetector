# envisionhgdetector/web/__init__.py
"""
Web interface module for EnvisionHG gesture detection.
"""

from .app import app, run_server

__all__ = ['app', 'run_server']
