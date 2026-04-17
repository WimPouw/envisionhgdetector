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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Model configuration matching best hyperparameter search result."""
    
    # Input settings
    seq_length: int = 25
    num_features: int = 92  # World landmarks: 23 × 4
    
    # Labels
    gesture_labels: Tuple[str, str] = ("Gesture", "Move")  # Motion classes (excluding NoGesture)
    all_labels: Tuple[str, str, str] = ("NoGesture", "Gesture", "Move")
    
    # Model architecture (from best config)
    conv_filters: Tuple[int, int, int] = (48, 96, 192)
    conv_kernel_size: int = 3
    pool_size: int = 2
    dense_units: int = 256
    dropout_rate: float = 0.36
    l2_weight: float = 0.0002
    
    # Preprocessing
    preprocessing: str = "basic"
    
    # Weights path (to be set by user or default location)
    weights_path: Optional[str] = None


# ============================================================================
# PREPROCESSING LAYERS
# ============================================================================

class BasicPreprocessing(layers.Layer):
    """
    Basic preprocessing - adds noise during training.
    This matches the training script's BasicPreprocessing.
    """
    def __init__(self, noise_stddev: float = 0.025, **kwargs):
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
    
    def get_config(self):
        config = super().get_config()
        config.update({'noise_stddev': self.noise_stddev})
        return config


class EnhancedPreprocessing(layers.Layer):
    """
    Enhanced preprocessing with derivatives and augmentation.
    """
    def __init__(
        self,
        noise_stddev: float = 0.005,
        jitter_sigma: float = 0.001,
        scale_range: Tuple[float, float] = (0.995, 1.005),
        drop_prob: float = 0.01,
        **kwargs
    ):
        super(EnhancedPreprocessing, self).__init__(**kwargs)
        self.noise_stddev = noise_stddev
        self.jitter_sigma = jitter_sigma
        self.scale_range = scale_range
        self.drop_prob = drop_prob

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_stddev': self.noise_stddev,
            'jitter_sigma': self.jitter_sigma,
            'scale_range': self.scale_range,
            'drop_prob': self.drop_prob
        })
        return config


# ============================================================================
# MODEL FACTORY
# ============================================================================

def make_model(
    weights_path: Optional[str] = None,
    num_features: int = 92,
    num_gesture_classes: int = 2,
    seq_length: int = 25,
    preprocessing: str = "basic",
    conv_filters: Tuple[int, int, int] = (48, 96, 192),
    dense_units: int = 256,
    dropout_rate: float = 0.36,
    l2_weight: float = 0.0002
) -> Model:
    """
    Create the gesture detection CNN model with residual blocks.
    
    Architecture matches best config from hyperparameter search:
    - Residual convolutional blocks with skip connections
    - Global average + max pooling
    - Hierarchical output (has_motion + gesture_probs)
    
    Args:
        weights_path: Path to load pretrained weights
        num_features: Number of input features (92 for world landmarks)
        num_gesture_classes: Number of gesture classes (2: Gesture, Move)
        seq_length: Sequence length (25 frames)
        preprocessing: "basic" or "enhanced"
        conv_filters: Tuple of filter counts for conv blocks
        dense_units: Dense layer units
        dropout_rate: Dropout rate
        l2_weight: L2 regularization weight
    
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=(seq_length, num_features), name="input")
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    if preprocessing == "enhanced":
        x = EnhancedPreprocessing()(inputs)
    else:
        x = BasicPreprocessing()(inputs)
    
    # ========================================================================
    # RESIDUAL CONVOLUTIONAL BLOCKS
    # ========================================================================
    for i, filters in enumerate(conv_filters):
        shortcut = x
        
        # Main path
        x = layers.Conv1D(
            filters,
            3,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(0.1)(x)
        
        # Downsample via pooling
        x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)
        
        # Shortcut path (1x1 conv to match dimensions)
        shortcut = layers.Conv1D(
            filters, 
            1, 
            strides=2, 
            padding="same",
            kernel_regularizer=regularizers.l2(l2_weight)
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
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(
        dense_units, 
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_weight)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # ========================================================================
    # HIERARCHICAL OUTPUT
    # ========================================================================
    # has_motion: binary (0=NoGesture, 1=Motion detected)
    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    
    # gesture_probs: softmax over motion types (Gesture, Move)
    gesture_probs = layers.Dense(
        num_gesture_classes,
        activation="softmax", 
        name="gesture_probs"
    )(x)
    
    # Combined output: [has_motion, gesture_prob_0, gesture_prob_1]
    outputs = layers.Concatenate(name="output")([has_motion, gesture_probs])

    model = Model(inputs, outputs)
    
    # Load weights if provided
    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"✓ Loaded weights from {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {weights_path}: {str(e)}")
    
    return model


# ============================================================================
# MODEL WRAPPER CLASS
# ============================================================================

class GestureModel:
    """
    Wrapper class for the gesture detection model.
    Handles model loading and inference.
    """
    
    def __init__(
        self, 
        config_or_path=None,
        num_features: Optional[int] = None,
        feature_set: Optional[str] = None
    ):
        """
        Initialize the model.
        
        Args:
            config_or_path: Either a Config object, a path string to weights, or None
            num_features: Number of input features (overrides config if provided)
            feature_set: "basic" (41), "extended" (61), or "world" (92)
        """
        # Handle different input types
        weights_path = None
        
        if config_or_path is None:
            # Use defaults
            feature_set = feature_set or "world"
        elif isinstance(config_or_path, str):
            # It's a path string
            weights_path = config_or_path
            feature_set = feature_set or "world"
        elif hasattr(config_or_path, 'weights_path'):
            # It's a Config object
            weights_path = config_or_path.weights_path
            feature_set = feature_set or getattr(config_or_path, 'feature_set', 'world')
            if num_features is None:
                num_features = getattr(config_or_path, 'num_original_features', None)
        else:
            raise ValueError(f"config_or_path must be None, a path string, or a Config object, got {type(config_or_path)}")
        
        # Set features based on feature_set if not explicitly provided
        if num_features is None:
            if feature_set == "world":
                num_features = 92
            elif feature_set == "extended":
                num_features = 61
            elif feature_set == "basic":
                num_features = 41
            else:
                num_features = 92  # Default to world
            
        self.num_features = num_features
        self.feature_set = feature_set
        self.seq_length = 25
        
        # Labels
        self.gesture_labels = ("Gesture", "Move")
        self.all_labels = ("NoGesture", "Gesture", "Move")
        
        # Build model
        self.model = make_model(
            weights_path=weights_path,
            num_features=num_features,
            num_gesture_classes=len(self.gesture_labels),
            seq_length=self.seq_length,
            preprocessing="basic"
        )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            features: Input features of shape (batch_size, seq_length, num_features)
            
        Returns:
            Model predictions of shape (batch_size, 3) where:
                - [:, 0] = has_motion probability
                - [:, 1] = Gesture probability (given motion)
                - [:, 2] = Move probability (given motion)
        """
        return self.model.predict(features, verbose=0)
    
    def predict_classes(
        self, 
        features: np.ndarray, 
        motion_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            features: Input features
            motion_threshold: Threshold for motion detection
            
        Returns:
            Array of class indices: 0=NoGesture, 1=Gesture, 2=Move
        """
        preds = self.predict(features)
        has_motion = preds[:, 0] > motion_threshold
        gesture_idx = np.argmax(preds[:, 1:], axis=1)
        
        # Combined class: 0=NoGesture, 1=Gesture, 2=Move
        return np.where(has_motion, gesture_idx + 1, 0)
    
    def predict_with_confidence(
        self, 
        features: np.ndarray,
        motion_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes with confidence scores.
        
        Args:
            features: Input features
            motion_threshold: Threshold for motion detection
            
        Returns:
            Tuple of (class_indices, confidence_scores)
        """
        preds = self.predict(features)
        has_motion = preds[:, 0] > motion_threshold
        motion_conf = preds[:, 0]
        
        gesture_idx = np.argmax(preds[:, 1:], axis=1)
        gesture_conf = np.max(preds[:, 1:], axis=1)
        
        # Combined class
        classes = np.where(has_motion, gesture_idx + 1, 0)
        
        # Confidence is motion_conf for NoGesture, gesture_conf * motion_conf for others
        confidence = np.where(
            has_motion,
            motion_conf * gesture_conf,
            1 - motion_conf
        )
        
        return classes, confidence


# ============================================================================
# CUSTOM LOSS AND METRICS (for training/evaluation)
# ============================================================================

def hierarchical_loss(y_true, y_pred):
    """
    Hierarchical loss for training.
    
    y_true format: [has_motion, gesture_onehot...]
        - NoGesture: [0, 0, 0] 
        - Gesture:   [1, 1, 0]
        - Move:      [1, 0, 1]
    """
    has_motion_true = y_true[:, :1]
    has_motion_pred = y_pred[:, :1]
    gesture_true = y_true[:, 1:]
    gesture_pred = y_pred[:, 1:]
    
    # Motion loss - standard BCE
    has_motion_loss = tf.keras.losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(has_motion_true, has_motion_pred)
    
    # Gesture loss - only for motion samples
    mask = tf.cast(y_true[:, 0] == 1, tf.float32)
    gesture_loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(gesture_true, gesture_pred, sample_weight=mask)
    
    return (has_motion_loss + gesture_loss) * 0.5


def custom_accuracy(y_true, y_pred):
    """
    Custom accuracy metric matching training.
    """
    motion_threshold = 0.5
    gesture_threshold = 0.5 
    
    y_pred_masked = tf.where(y_pred[:, :1] >= motion_threshold, y_pred, 0.0)
    y_pred_binary = tf.where(y_pred_masked >= gesture_threshold, 1.0, 0.0)
    
    return tf.keras.metrics.categorical_accuracy(y_true[:, 1:], y_pred_binary[:, 1:])