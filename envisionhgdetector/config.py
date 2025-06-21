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
        # CNN model weights path
        try:
            # Using importlib.resources (Python 3.9+)
            self.weights_path = str(files('envisionhgdetector').joinpath('model/model_weights_20250224_103340.h5'))
        except:
            # Fallback for older Python versions or if file doesn't exist
            try:
                # Or using pkg_resources (older Python versions)
                # self.weights_path = resource_filename('envisionhgdetector', 'model/SAGAplus_gesturenogesture_trained_binaryCNNmodel_weightsv1.h5')
                self.weights_path = str(files('envisionhgdetector').joinpath('model/model_weights_20250224_103340.h5'))
            except:
                # Final fallback - check if file exists in expected locations
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), 'model', 'model_weights_20250224_103340.h5'),
                    'model_weights_20250224_103340.h5'
                ]
                self.weights_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        self.weights_path = path
                        break
        
        # LightGBM model weights path
        try:
            # Using importlib.resources (Python 3.9+)
            self.lightgbm_weights_path = str(files('envisionhgdetector').joinpath('model/lightgbm_gesture_model_v1.pkl'))
            # Check if file actually exists
            if not os.path.exists(self.lightgbm_weights_path):
                self.lightgbm_weights_path = None
        except:
            # Fallback - check if file exists in expected locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'model', 'lightgbm_gesture_model_v1.pkl'),
                'lightgbm_gesture_model_v1.pkl'
            ]
            self.lightgbm_weights_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.lightgbm_weights_path = path
                    break
    
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