import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import List, Literal
import numpy as np
from ..state import CNN_B_Config, CNN_Config

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
    def __init__(self, config: CNN_B_Config | CNN_Config, **kwargs):
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

def create_windows(self, features: List[List[float]], seq_length: int, stride: int) -> np.ndarray:
        """Creates sliding windows from feature sequences (CNN only)."""
        windows = []
        if len(features) < seq_length:
            return np.array([])
        for i in range(0, len(features) - seq_length + 1, stride):
            windows.append(features[i:i + seq_length])
        return np.array(windows)

def make_model(config: CNN_B_Config | CNN_Config, type: Literal["multi", "binary"]) -> Model:
    """
    Create the gesture detection CNN model with residual blocks.
    
    Architecture matches best config from hyperparameter search:
    - Residual convolutional blocks with skip connections
    - Global average + max pooling
    - Binary sigmoid output (0: NoGesture, 1: Gesture) OR
    - Multi-class output (0: NoGesture, 1: Gesture, 2: Move) with hierarchical structure
    
    Args:
        config: CNN_B_Config or CNN_Config object containing model parameters.
        type: Type of output (binary or multi-class).
    
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

    if type == "multi": # hierarchical output
        has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
            
        # gesture_probs: softmax over motion types (Gesture, Move)
        if not hasattr(config, 'gesture_labels'):
            raise ValueError("Config object must have 'gesture_labels' attribute for multi-class output.")
        gesture_probs = layers.Dense(
            len(config.gesture_labels),
            activation="softmax", 
            name="gesture_probs"
        )(x)
        outputs = layers.Concatenate(name="output")([has_motion, gesture_probs])
    elif type == "binary":
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    else:
        raise ValueError(f"Unknown model type: {type}. Use 'multi' or 'binary'.")
    
    model = Model(inputs, outputs)
    
    # Load weights if provided
    if config.weights_path:
        try:
            model.load_weights(config.weights_path)
            print(f"✓ Loaded weights from {config.weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {config.weights_path}: {str(e)}")
    
    return model

