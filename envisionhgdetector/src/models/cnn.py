# envisionhgdetector/envisionhgdetector/model.py
"""
Gesture detection CNN model.
Architecture matches best performing config from hyperparameter search.

Best model: World landmarks (92 features) with residual CNN blocks.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional, Tuple
from preprocessing import BasicPreprocessing, EnhancedPreprocessing
import numpy as np

# ============================================================================
# MODEL FACTORY
# ============================================================================

def make_model(
    config: dict,
    num_gesture_classes: int
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
    seq_length = config.get("data").get("seq_length")
    num_features = config.get("data").get("num_features")
    inputs = layers.Input(shape=(seq_length, num_features), name="input")
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    preprocessing = config.get("model").get("preprocessing")
    if preprocessing == "enhanced":
        x = EnhancedPreprocessing()(inputs)
    else:
        x = BasicPreprocessing()(inputs)
    
    # ========================================================================
    # RESIDUAL CONVOLUTIONAL BLOCKS
    # ========================================================================
    conv_filters = config.get("model").get("conv_filters")
    l2_weight = config.get("model").get("l2_weight")
    dropout_rate = config.get("model").get("dropout_rate")
    conv_kernel_size = config.get("model").get("conv_kernel_size")
    spatial_dropout_rate = config.get("model").get("spatial_dropout_rate")
    pool_size = config.get("model").get("pool_size")
    dense_units = config.get("model").get("dense_units")
    
    for _, filters in enumerate(conv_filters):
        shortcut = x
        
        # Main path
        x = layers.Conv1D(
            filters,
            kernel_size=conv_kernel_size,
            padding="same",
            kernel_regularizer=regularizers.l2(l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(spatial_dropout_rate)(x)
        
        # Downsample via pooling
        x = layers.MaxPooling1D(pool_size=pool_size, strides=2, padding='same')(x)
        
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

    model = Model(inputs, outputs, name="residual_cnn_gesture")
    
    # Load weights if provided
    weights_path = config.get("model").get("file_path")
    try:
        model.load_weights(weights_path)
        print(f"✓ Loaded weights from {weights_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {weights_path}: {str(e)}")
    
    return model

# ============================================================================
# MODEL WRAPPER CLASS
# ============================================================================

class CNNGestureModel:
    """
    Wrapper class for the gesture detection model.
    Handles model loading and inference.
    """
    def __init__(self, config: dict):
        """
        Initialize the model.
        
        Args:
            config_or_path: Either a Config object, a path string to weights, or None
            num_features: Number of input features (overrides config if provided)
            feature_set: "basic" (41), "extended" (61), or "world" (92)
        """
        
        # Labels
        self.gesture_labels = config.get("data").get("gesture_labels")
        self.class_labels = config.get("data").get("class_labels")
        
        # Build model
        self.model = make_model(
            config=config,
            num_gesture_classes=len(self.gesture_labels)
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