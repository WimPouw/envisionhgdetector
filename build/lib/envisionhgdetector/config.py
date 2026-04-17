# envisionhgdetector/config.py
"""
Configuration for the gesture detection system.
Supports three feature sets: basic (41), extended (61), world (92).
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from importlib.resources import files
import os

@dataclass
class Config:
    """Configuration for the gesture detection system."""
    
    # ========================================================================
    # MODEL CONFIGURATION - CHANGE THESE TO SWITCH MODELS
    # ========================================================================
    
    # Feature set: "basic" (41), "extended" (61), or "world" (92)
    feature_set: str = "world"  # Best performing model
    
    # CNN model filename (in envisionhgdetector/model/)
    # Set to None to use default based on feature_set
    cnn_model_filename: Optional[str] = "cnn_default.h5"
    
    # LightGBM model filename (in envisionhgdetector/model/)
    lightgbm_model_filename: Optional[str] = "lightgbm_default.pkl"
    
    # ========================================================================
    # GESTURE LABELS
    # ========================================================================
    gesture_labels: Tuple[str, ...] = ("Gesture", "Move")
    all_labels: Tuple[str, ...] = ("NoGesture", "Gesture", "Move")
    undefined_gesture_label: str = "Undefined"
    stationary_label: str = "NoGesture"
    
    # ========================================================================
    # MODEL PARAMETERS
    # ========================================================================
    seq_length: int = 25  # Window size for classification
    
    # Number of input features (set automatically based on feature_set)
    num_original_features: int = 92
    
    # Model architecture (for world landmarks - best config)
    conv_filters: Tuple[int, int, int] = (48, 96, 192)
    dense_units: int = 256
    dropout_rate: float = 0.36
    l2_weight: float = 0.0002
    preprocessing: str = "basic"
    
    # ========================================================================
    # DEFAULT THRESHOLDS
    # ========================================================================
    default_motion_threshold: float = 0.7
    default_gesture_threshold: float = 0.7
    default_min_gap_s: float = 0.5
    default_min_length_s: float = 0.5
    
    # Internal: resolved paths (set in __post_init__)
    weights_path: Optional[str] = field(default=None, init=False)
    lightgbm_weights_path: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        """Setup paths after initialization."""
        
        # Set number of features based on feature_set
        if self.feature_set == "world":
            self.num_original_features = 92
        elif self.feature_set == "extended":
            self.num_original_features = 61
        else:  # basic
            self.num_original_features = 41
        
        # Set default model filename based on feature_set if not specified
        if self.cnn_model_filename is None:
            self.cnn_model_filename = "R2_CNN_world_best_config18.h5"
           
        # ====================================================================
        # CNN MODEL PATH
        # ====================================================================
        self.weights_path = self._resolve_model_path(self.cnn_model_filename)
        
        # ====================================================================
        # LIGHTGBM MODEL PATH
        # ====================================================================
        if self.lightgbm_model_filename:
            self.lightgbm_weights_path = self._resolve_model_path(self.lightgbm_model_filename)
    
    def _resolve_model_path(self, filename: str) -> Optional[str]:
        """
        Resolve model path from filename.
        Tries multiple locations in order of preference.
        """
        if filename is None:
            return None
            
        # Try importlib.resources first (Python 3.9+)
        try:
            path = str(files('envisionhgdetector').joinpath(f'model/{filename}'))
            if os.path.exists(path):
                return path
        except:
            pass
        
        # Fallback - check common locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'model', filename),
            os.path.join(os.path.dirname(__file__), filename),
            filename,
            os.path.join('model', filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return expected path even if doesn't exist (for error messages)
        return os.path.join(os.path.dirname(__file__), 'model', filename)
    
    def get_model_path(self, model_type: str = "cnn") -> Optional[str]:
        """Get the appropriate model path based on model type."""
        if model_type.lower() == "lightgbm":
            return self.lightgbm_weights_path
        elif model_type.lower() == "cnn":
            return self.weights_path
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def is_model_available(self, model_type: str = "cnn") -> bool:
        """Check if a model is available."""
        path = self.get_model_path(model_type)
        return path is not None and os.path.exists(path)
    
    @property
    def available_models(self):
        """Get list of available models."""
        models = []
        if self.is_model_available("cnn"):
            models.append("cnn")
        if self.is_model_available("lightgbm"):
            models.append("lightgbm")
        return models
    
    @property
    def default_thresholds(self):
        """Return default threshold parameters as dictionary."""
        return {
            'motion_threshold': self.default_motion_threshold,
            'gesture_threshold': self.default_gesture_threshold,
            'min_gap_s': self.default_min_gap_s,
            'min_length_s': self.default_min_length_s
        }
    
    def __repr__(self):
        """Pretty print configuration."""
        return (
            f"Config(\n"
            f"  feature_set='{self.feature_set}',\n"
            f"  num_features={self.num_original_features},\n"
            f"  seq_length={self.seq_length},\n"
            f"  cnn_model='{self.cnn_model_filename}',\n"
            f"  lightgbm_model='{self.lightgbm_model_filename}',\n"
            f"  cnn_available={self.is_model_available('cnn')},\n"
            f"  lightgbm_available={self.is_model_available('lightgbm')}\n"
            f")"
        )


# Default configuration instance
DEFAULT_CONFIG = Config()