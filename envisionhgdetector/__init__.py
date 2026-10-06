"""
EnvisionHGDetector: Hand Gesture Detection Package
Supports CNN, LightGBM, and Combined models for gesture detection.
"""

from .models.cnn.model_cnn import GestureModel
from .models.cnn.model_cnn_b import GestureModel as BinaryGestureModel
from .models.lightgbm.model_lightgbm import LightGBMGestureModel

from .detectors.detector import GestureDetector
from .detectors.realtime_detection import RealtimeGestureDetector
from .combined_detection import CombinedGestureDetector

from .rendering.label_video import label_video, label_video_batch

__version__ = "3.1.0"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele"

# import *
__all__ = [
    # Main detector
    "GestureDetector",
    "RealtimeGestureDetector",
    "CombinedGestureDetector",
    
    # Individual models
    "BinaryGestureModel",     # Binary CNN model
    "GestureModel",           # CNN model
    "LightGBMGestureModel",   # LightGBM model
]