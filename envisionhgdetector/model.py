# envisionhgdetector/envisionhgdetector/model.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional
import numpy as np
from .config import Config

class Preprocessing(layers.Layer):
    """Custom preprocessing layer for feature normalization and augmentation."""
    
    def __init__(self) -> None:
        super(Preprocessing, self).__init__(name="preprocessing")

    def call(self, x: tf.Tensor,
            training: Optional[bool] = None,
            mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        features = x

        # Center features
        features = features - tf.reduce_mean(features, axis=-2, keepdims=True)

        # Compute time derivatives
        t_deriv = features[:, 1:] - features[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])

        # Compute second derivatives
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])

        # Compute standard deviations
        features_std = tf.math.reduce_std(features, axis=-2, keepdims=True)
        t_deriv_std = tf.math.reduce_std(t_deriv, axis=-2, keepdims=True)
        t_deriv_std_2 = tf.math.reduce_std(t_deriv_2, axis=-2, keepdims=True)

        # Concatenate features
        features = tf.concat([
            features,
            t_deriv,
            t_deriv_2,
            tf.broadcast_to(features_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std_2, tf.shape(features))
        ], axis=-1)

        # Optional training augmentation
        if training:
            features = features + tf.random.normal(tf.shape(features), mean=0.0, stddev=0.01)

        # Final normalization
        features = (features - tf.reduce_mean(features, axis=-1, keepdims=True)) / (
            tf.math.reduce_std(features, axis=-1, keepdims=True) + 1e-8)

        return features

def make_model(weights_path: Optional[str] = None) -> Model:
    """
    Create and load the gesture detection model.
    
    Args:
        weights_path: Optional path to model weights file
        
    Returns:
        Loaded TensorFlow model ready for inference
    """
    config = Config()
    
    # Input layer
    seq_input = layers.Input(
        shape=(config.seq_length, config.num_original_features),
        dtype=tf.float32, 
        name="input"
    )
    
    # Model architecture
    x = seq_input
    x = Preprocessing()(x)
    
    # Convolutional layers
    x = layers.Conv1D(32, 3, strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    x = layers.Conv1D(64, 3, strides=1, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    x = layers.Conv1D(128, 3, strides=1, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x_0 = layers.Dropout(0.01)(x)

    # Output layers
    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x_0)
    gesture_probs = layers.Dense(len(config.gesture_labels),
                               activation="softmax", name="gesture_probs")(x_0)
    output = layers.Concatenate()([has_motion, gesture_probs])

    # Create model
    model = Model(seq_input, output)
    
    # Load weights if provided
    if weights_path:
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from {weights_path}: {str(e)}")
    
    return model

class GestureModel:
    """
    Wrapper class for the gesture detection model.
    Handles model loading and inference.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the model with optional custom configuration."""
        self.config = config or Config()
        self.model = make_model(self.config.weights_path)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Run inference on input features.
        
        Args:
            features: Input features of shape (batch_size, seq_length, num_features)
            
        Returns:
            Model predictions
        """
        return self.model.predict(features, verbose=0)