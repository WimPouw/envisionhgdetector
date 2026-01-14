# envisionhgdetector/config.py

from dataclasses import dataclass
from typing import Tuple
from importlib.resources import files  # if using Python 3.9+
# or from pkg_resources import resource_filename  # for older Python versions
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
    default_motion_threshold: float = 0.7
    default_gesture_threshold: float = 0.7
    default_min_gap_s: float = 0.5
    default_min_length_s: float = 0.5
    
    def __post_init__(self):
        """Setup paths after initialization."""
        self.weights_path = self._find_model_path(
            'model/model_weights_20250224_103340.h5',
            'model_weights_20250224_103340.h5'
        )
        self.lightgbm_weights_path = self._find_model_path(
            'model/lightgbm_gesture_model_v1.pkl',
            'lightgbm_gesture_model_v1.pkl'
        )

    def _find_model_path(self, resource_path: str, filename: str) -> str:
        """
        Find model path using importlib.resources with fallbacks.

        Args:
            resource_path: Path relative to package (e.g., 'model/weights.h5')
            filename: Just the filename for fallback search

        Returns:
            Absolute path to model file, or None if not found
        """
        # Try importlib.resources first (Python 3.9+)
        try:
            path = str(files('envisionhgdetector').joinpath(resource_path))
            if os.path.exists(path):
                return path
        except (TypeError, FileNotFoundError, ModuleNotFoundError):
            pass  # Fall through to manual search

        # Fallback - check common locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), resource_path),
            os.path.join(os.path.dirname(__file__), 'model', filename),
            filename
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None
    
    def get_model_path(self, model_type: str):
        """Get the appropriate model path based on model type."""
        if model_type.lower() == "lightgbm":
            return self.lightgbm_weights_path
        elif model_type.lower() == "cnn":
            return self.weights_path
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def is_model_available(self, model_type: str) -> bool:
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