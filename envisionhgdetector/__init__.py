"""
EnvisionHGDetector: Hand Gesture Detection Package
"""

from .config import Config
from .detector import GestureDetector
from .model import GestureModel, make_model

__version__ = "0.0.4.8"
__author__ = "Wim Pouw"
__email__ = "wim.pouw@donders.ru.nl"

# Make key classes available at package level
__all__ = [
    'Config',
    'GestureDetector',
    'GestureModel',
    'make_model'
]

# Example usage in docstring
__doc__ = """
EnvisionHGDetector is a package for detecting hand gestures in videos.

Basic usage:
    from envisionhgdetector import GestureDetector
    
    detector = GestureDetector()
    results = detector.process_folder(
        input_folder="path/to/videos",
        output_folder="path/to/output"
    )
"""