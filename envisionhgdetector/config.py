
from dataclasses import dataclass
from typing import Tuple
import os

@dataclass
class Config:
    """Configuration for the gesture detection system."""
    
    # Model configuration
    gesture_labels: Tuple[str, ...] = ("Gesture", "Move")
    undefined_gesture_label: str = "Undefined"
    stationary_label: str = "NoGesture"
    seq_length: int = 25  # Window size for classification
    num_original_features: int = 29  # Number of input features
    
    # Default thresholds (can be overridden in detector)
    default_motion_threshold: float = 0.8
    default_gesture_threshold: float = 0.8
    default_min_gap_s: float = 0.3
    default_min_length_s: float = 0.3
    
    def __post_init__(self):
        """Setup paths after initialization."""
        # Get the package root directory
        package_root = os.path.dirname(os.path.dirname(__file__))
        
        # Set up model weights path
        self.weights_path = os.path.join(
            package_root,
            "model",
            "SAGAplus_gesturenogesture_trained_binaryCNNmodel_weightsv1.h5"
        )
    
    @property
    def default_thresholds(self):
        """Return default threshold parameters as dictionary."""
        return {
            'motion_threshold': self.default_motion_threshold,
            'gesture_threshold': self.default_gesture_threshold,
            'min_gap_s': self.default_min_gap_s,
            'min_length_s': self.default_min_length_s
        }

# envisionhgdetector/envisionhgdetector/__init__.py

"""
EnvisionHGDetector: Head Gesture Detection Package
"""

from .config import Config
from .detector import GestureDetector

__version__ = "0.0.1"
__author__ = "Wim Pouw"
__email__ = "wim.pouw@donders.ru.nl"

# Make key classes available at package level
__all__ = ['Config', 'GestureDetector']

# Example usage in docstring
__doc__ = """
EnvisionHGDetector is a package for detecting hand gestures in videos.

Basic usage:
    from envisionhgdetector import GestureDetector
    
    detector = GestureDetector()
    results = detector.process_folder(
        video_folder="path/to/videos",
        output_folder="path/to/output"
    )
"""