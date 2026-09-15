# envisionhgdetector/envisionhgdetector/model.py
"""
Gesture detection CNN model.
Architecture matches best performing config from hyperparameter search.

Best model: World landmarks (92 features) with residual CNN blocks.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional, Tuple
import numpy as np

from .default_config import CNN_B_Config

# ============================================================================
# CONFIGURATION
# ============================================================================
'''
# class Config:
#     """Model configuration matching best hyperparameter search result."""
    
#     # Input settings
#     seq_length: int = 25
#     num_features: int = 92  # World landmarks: 23 × 4
    
#     # Model architecture (from best config)
#     conv_filters: Tuple[int, int, int] = (48, 96, 192)
#     conv_kernel_size: int = 3
#     pool_size: int = 2
#     dense_units: int = 256
#     dropout_rate: float = 0.36
#     l2_weight: float = 0.0002
    
#     # Preprocessing
#     preprocessing: str = "basic"
    
#     # Weights path (to be set by user or default location)
#     weights_path: Optional[str] = None
'''


# ============================================================================
# PREPROCESSING LAYERS
# ============================================================================

class BasicPreprocessing(layers.Layer):
    """
    Basic preprocessing - adds noise during training.
    This matches the training script's BasicPreprocessing.
    """
    def __init__(self, noise_stddev: float, **kwargs):
        super(BasicPreprocessing, self).__init__(**kwargs)
        self.noise_stddev = noise_stddev
        
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.noise_stddev,
                dtype=tf.float32
            )
            inputs = inputs + noise
        return inputs
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({'noise_stddev': self.noise_stddev})
    #     return config


class EnhancedPreprocessing(layers.Layer):
    """
    Enhanced preprocessing with derivatives and augmentation.
    """
    def __init__(self, config: CNN_B_Config, **kwargs):
        super(EnhancedPreprocessing, self).__init__(**kwargs)
        self.noise_stddev = config.noise_stddev
        self.jitter_sigma = config.jitter_sigma
        self.scale_range = config.scale_range
        self.drop_prob = config.drop_prob

    def call(self, inputs, training=None):
        x = inputs
        
        # Center features
        x = x - tf.reduce_mean(x, axis=-2, keepdims=True)
        
        # Compute derivatives
        t_deriv = x[:, 1:] - x[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])
        
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])
        
        # Concatenate features with derivatives
        x = tf.concat([x, t_deriv, t_deriv_2], axis=-1)
        
        if training:
            # Add noise
            x = x + tf.random.normal(tf.shape(x), stddev=self.noise_stddev)
            
            # Random scaling
            scale = tf.random.uniform([], self.scale_range[0], self.scale_range[1])
            x = x * scale
            
            # Random frame drop
            mask = tf.cast(tf.random.uniform(tf.shape(x)[:2]) > self.drop_prob, x.dtype)
            x = x * mask[:, :, tf.newaxis]
        
        # Normalize
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True) + 1e-8
        x = (x - mean) / std
        
        return x
    
    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'noise_stddev': self.noise_stddev,
    #         'jitter_sigma': self.jitter_sigma,
    #         'scale_range': self.scale_range,
    #         'drop_prob': self.drop_prob
    #     })
    #     return config


# ============================================================================
# MODEL FACTORY
# ============================================================================

def make_model(config: CNN_B_Config) -> Model:
    """
    Create the gesture detection CNN model with residual blocks.
    
    Architecture matches best config from hyperparameter search:
    - Residual convolutional blocks with skip connections
    - Global average + max pooling
    - Binary sigmoid output (0: NoGesture, 1: Gesture)
    
    Args:
        config: CNN_B_Config object containing model parameters.
    
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=(config.seq_length, config.num_features), name="input")
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    # if config.preprocessing == "enhanced":
    #     x = EnhancedPreprocessing(config)(inputs)
    # else:
    #     x = BasicPreprocessing(config.noise_stddev)(inputs)
    # TODO -- training only has basic -- fix that -- then fix this
    x = BasicPreprocessing(config.noise_stddev)(inputs)  # Using basic preprocessing for now
    
    # ========================================================================
    # RESIDUAL CONVOLUTIONAL BLOCKS
    # ========================================================================
    for i, filters in enumerate(config.conv_filters):
        shortcut = x
        
        # Main path
        x = layers.Conv1D(
            filters,
            config.conv_kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(config.l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(config.spatial_dropout_rate)(x)
        
        # Downsample via pooling
        x = layers.MaxPooling1D(pool_size=config.pool_size, strides=2, padding='same')(x)
        
        # Shortcut path (1x1 conv to match dimensions)
        shortcut = layers.Conv1D(
            filters, 
            1, 
            strides=2, 
            padding="same",
            kernel_regularizer=regularizers.l2(config.l2_weight)
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
        # Merge
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
    
    # ========================================================================
    # POOLING AND HEAD
    # ========================================================================
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])
    x = layers.Dropout(config.dropout_rate)(x)
    
    x = layers.Dense(
        config.dense_units, 
        activation="relu",
        kernel_regularizer=regularizers.l2(config.l2_weight)
    )(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # ========================================================================
    # BINARY OUTPUT
    # ========================================================================
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inputs, outputs)
    
    # Load weights if provided
    if config.weights_path:
        try:
            model.load_weights(config.weights_path)
            print(f"✓ Loaded weights from {config.weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {config.weights_path}: {str(e)}")
    
    return model


# ============================================================================
# MODEL WRAPPER CLASS
# ============================================================================

class GestureModel:
    """
    Wrapper class for the gesture detection model.
    Handles model loading and inference.
    """
    def __init__(self, config: CNN_B_Config):
        """
        Initialize the model.
        
        Args:
            config: CNN_B_Config
        """
        print('got these')
        print(config)

        # TODO handle this beforehand
        '''# Handle different input types
        # weights_path = None
        
        # if config_or_path is None:
        #     # Use defaults
        #     feature_set = feature_set or "world"
        # elif isinstance(config_or_path, str):
        #     # It's a path string
        #     weights_path = config_or_path
        #     feature_set = feature_set or "world"
        # elif hasattr(config_or_path, 'weights_path'):
        #     # It's a Config object
        #     weights_path = config_or_path.weights_path
        #     feature_set = feature_set or getattr(config_or_path, 'feature_set', 'world')
        #     if num_features is None:
        #         num_features = getattr(config_or_path, 'num_original_features', None)
        # else:
        #     raise ValueError(f"config_or_path must be None, a path string, or a Config object, got {type(config_or_path)}")
        
        # Set features based on feature_set if not explicitly provided
        # if num_features is None:
        #     if feature_set == "world":
        #         num_features = 92
        #     elif feature_set == "extended":
        #         num_features = 61
        #     elif feature_set == "basic":
        #         num_features = 41
        #     else:
        #         num_features = 92  # Default to world
            
        # self.num_features = num_features
        # self.feature_set = feature_set
        # self.seq_length = 25

        # print(f"Initializing CNN B with weights_path={weights_path}, ")
        '''
        # Build model
        self.model = make_model(config=config)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            features: Input features of shape (batch_size, seq_length, num_features)
            
        Returns:
            Model predictions of shape (batch_size, 1), containing
            Gesture probabilities.
        """
        return self.model.predict(features, verbose=0)
    
    def predict_classes(self, features: np.ndarray, motion_threshold: float) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            features: Input features
            motion_threshold: Threshold for motion detection
            
        Returns:
            Array of class indices: 0=NoGesture, 1=Gesture
        """
        preds = self.predict(features)
        return (preds[:, 0] > motion_threshold).astype(np.int64)
    
    def predict_with_confidence(self, features: np.ndarray, motion_threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes with confidence scores.
        
        Args:
            features: Input features
            motion_threshold: Threshold for motion detection
            
        Returns:
            Tuple of (class_indices, confidence_scores)
        """
        preds = self.predict(features)
        classes = (preds[:, 0] > motion_threshold).astype(np.int64)
        # condition, if true, else
        confidence = np.where(classes == 1, preds[:, 0], 1.0 - preds[:, 0])
        return classes, confidence


# ============================================================================
# CUSTOM LOSS AND METRICS (for training/evaluation)
# ============================================================================

def hierarchical_loss(y_true, y_pred):
    """
    Binary cross-entropy for scalar labels: 0=NoGesture, 1=Gesture.
    """
    return tf.keras.losses.BinaryCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(y_true, y_pred)


def custom_accuracy(y_true, y_pred):
    """
    Custom accuracy metric matching training.
    """
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)