"""
EnvisionHGDetector: Hand Gesture Detection Package
Supports CNN, LightGBM, and Combined models for gesture detection.
"""

from .detector import GestureDetector, RealtimeGestureDetector
from .combined_detection import CombinedGestureDetector
from .cnn.model_cnn import GestureModel
from .cnn.model_cnn_b import GestureModel as BinaryGestureModel
from .lightgbm.model_lightgbm import LightGBMGestureModel

__version__ = "3.0.1"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele"

__all__ = [
    # Main detector
    "GestureDetector",
    "RealtimeGestureDetector",
    "CombinedGestureDetector",
    
    # Configuration
    "Config",
    "CombinedConfig",
    
    # Individual models
    "BinaryGestureModel",     # Binary CNN model
    "GestureModel",           # CNN model
    "LightGBMGestureModel",   # LightGBM model
    
    # Convenience functions
    "load_combined_model",
]