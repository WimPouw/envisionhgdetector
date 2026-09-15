"""
EnvisionHGDetector: Hand Gesture Detection Package
Supports CNN, LightGBM, and Combined models for gesture detection.
"""

from .config import Config
from .detector import GestureDetector, RealtimeGestureDetector
from .model_cnn import GestureModel
from .model_cnn_b import GestureModel as BinaryGestureModel
from .model_lightgbm import LightGBMGestureModel
from .model_combined import CombinedGestureModel, CombinedConfig, load_combined_model

__version__ = "3.0.1"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele"

__all__ = [
    # Main detector
    "GestureDetector",
    "RealtimeGestureDetector",
    
    # Configuration
    "Config",
    "CombinedConfig",
    
    # Individual models
    "BinaryGestureModel",     # Binary CNN model
    "GestureModel",           # CNN model
    "LightGBMGestureModel",   # LightGBM model
    "CombinedGestureModel",   # Combined CNN + LightGBM
    
    # Convenience functions
    "load_combined_model",
]