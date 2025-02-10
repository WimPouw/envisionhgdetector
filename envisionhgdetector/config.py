from dataclasses import dataclass
from typing import Tuple
from importlib.resources import files  # if using Python 3.9+
# or from pkg_resources import resource_filename  # for older Python versions

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
    default_motion_threshold: float = 0.5
    default_gesture_threshold: float = 0.8
    default_min_gap_s: float = 0.5
    default_min_length_s: float = 0.5
    
    def __post_init__(self):
        """Setup paths after initialization."""
        # Using importlib.resources (Python 3.9+)
        self.weights_path = str(files('envisionhgdetector').joinpath('model/model_weights_20250210_042008.h5'))
        
        # Or using pkg_resources (older Python versions)
        # self.weights_path = resource_filename('envisionhgdetector', 'model/SAGAplus_gesturenogesture_trained_binaryCNNmodel_weightsv1.h5')
    
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
EnvisionHGDetector: Hand Gesture Detection Package
"""

from .config import Config
from .detector import GestureDetector

__version__ = "0.0.5.1"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Gerard de Melo, Babajide Owoyele"
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