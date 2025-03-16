# envisionhgdetector/envisionhgdetector/model.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional
import numpy as np
from .config import Config
from typing import Tuple

class EnhancedPreprocessing(layers.Layer):
    def __init__(
        self,
        time_warp_range: Tuple[float, float] = (0.9, 1.1),    # Less aggressive time warping
        rotation_range: Tuple[float, float] = (-0.05, 0.05),  # Smaller rotation range
        jitter_sigma: float = 0.005,                          # Smaller jitter for normalized distances
        drop_prob: float = 0.03,                              # Lower drop probability
        noise_stddev: float = 0.01,                           # Lower noise for normalized features
        scale_range: Tuple[float, float] = (0.98, 1.02),      # Smaller scale range
        mask_max_size: int = 2,                               # Shorter mask length
        original_feature_dim: int = 29                        # Dimension of original features
    ) -> None:
        super(EnhancedPreprocessing, self).__init__(name="enhanced_preprocessing")
        self.time_warp_range = time_warp_range
        self.rotation_range = rotation_range
        self.jitter_sigma = jitter_sigma
        self.drop_prob = drop_prob
        self.noise_stddev = noise_stddev
        self.scale_range = scale_range
        self.mask_max_size = mask_max_size
        self.original_feature_dim = original_feature_dim

    def time_warp(self, features: tf.Tensor) -> tf.Tensor:
        """Apply random temporal warping."""
        warp = tf.random.uniform([], self.time_warp_range[0], self.time_warp_range[1])
        seq_len = tf.shape(features)[1]
        warped = tf.image.resize(
            features[:, :, :, tf.newaxis], 
            [seq_len, tf.shape(features)[2]]
        )[:, :, :, 0]
        return warped

    def add_position_jitter(self, features: tf.Tensor) -> tf.Tensor:
        """Add random jitter to positions."""
        jitter = tf.random.normal(tf.shape(features), mean=0.0, stddev=self.jitter_sigma)
        return features + jitter

    def random_frame_drop(self, features: tf.Tensor) -> tf.Tensor:
        """Randomly drop frames to simulate tracking failures."""
        mask = tf.random.uniform(tf.shape(features)[:2]) > self.drop_prob
        mask = tf.cast(mask, features.dtype)
        return features * mask[:, :, tf.newaxis]

    def time_masking(self, features: tf.Tensor) -> tf.Tensor:
        """Apply random time masking."""
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(features)[1]
        feature_dim = tf.shape(features)[2]
        
        mask_size = tf.random.uniform([], 1, self.mask_max_size, dtype=tf.int32)
        starts = tf.random.uniform([batch_size], 0, seq_len - mask_size, dtype=tf.int32)
        
        mask = tf.ones([batch_size, seq_len, feature_dim])
        
        for i in range(batch_size):
            start = starts[i]
            indices = tf.range(start, start + mask_size)
            mask_updates = tf.zeros([mask_size, feature_dim])
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.stack([tf.repeat(i, mask_size), indices], axis=1),
                mask_updates
            )
        
        return features * mask

    def compute_derivatives(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute first and second derivatives."""
        # First derivative
        t_deriv = features[:, 1:] - features[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])

        # Second derivative
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])
        
        return t_deriv, t_deriv_2

    def compute_statistics(
        self, 
        features: tf.Tensor, 
        t_deriv: tf.Tensor, 
        t_deriv_2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute statistical features."""
        features_std = tf.math.reduce_std(features, axis=-2, keepdims=True)
        t_deriv_std = tf.math.reduce_std(t_deriv, axis=-2, keepdims=True)
        t_deriv_std_2 = tf.math.reduce_std(t_deriv_2, axis=-2, keepdims=True)
        
        return features_std, t_deriv_std, t_deriv_std_2

    def normalize_features(self, features: tf.Tensor) -> tf.Tensor:
        """Apply feature normalization only to original features.
        
        Args:
            features: Input tensor of shape [batch_size, sequence_length, feature_dim]
        """
        # Split features into original and derived
        original_features = features[:, :, :self.original_feature_dim]
        derived_features = features[:, :, self.original_feature_dim:]
        
        # Normalize only the original features
        mean = tf.reduce_mean(original_features, axis=-1, keepdims=True)
        std = tf.math.reduce_std(original_features, axis=-1, keepdims=True) + 1e-8
        normalized_original = (original_features - mean) / std
        
        # Concatenate back
        return tf.concat([normalized_original, derived_features], axis=-1)

    def call(
        self, 
        x: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        features = x
        
        # Split features for centering
        original_features = features[:, :, :self.original_feature_dim]
        derived_features = features[:, :, self.original_feature_dim:]
        
        # Center only the original features
        centered_original = original_features - tf.reduce_mean(original_features, axis=-2, keepdims=True)
        
        # Recombine features
        features = tf.concat([centered_original, derived_features], axis=-1)
        
        # Compute derivatives and statistics on ALL features
        t_deriv, t_deriv_2 = self.compute_derivatives(features)
        features_std, t_deriv_std, t_deriv_std_2 = self.compute_statistics(
            features, t_deriv, t_deriv_2
        )
        
        # Concatenate all features
        features = tf.concat([
            features,
            t_deriv,
            t_deriv_2,
            tf.broadcast_to(features_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std_2, tf.shape(features))
        ], axis=-1)
        
        if training:
            # Apply augmentations to ALL features
            features = self.time_warp(features)
            features = self.add_position_jitter(features)
            features = self.random_frame_drop(features)
            
            # Add noise to ALL features
            features = features + tf.random.normal(
                tf.shape(features), 
                mean=0.0, 
                stddev=self.noise_stddev
            )
            
            # Random scaling for ALL features
            scale = tf.random.uniform(
                [], 
                self.scale_range[0], 
                self.scale_range[1], 
                dtype=tf.float32
            )
            features = features * scale
            
            # Apply time masking to ALL features
            features = self.time_masking(features)
        
        # Final normalization - only normalize the original features
        features = self.normalize_features(features)
        
        return features

def make_model(weights_path: Optional[str] = None) -> Model:
    seq_input = layers.Input(
        shape=(Config.seq_length, Config.num_original_features),
        dtype=tf.float32, name="input"
    )
    x = seq_input
    
    # Replace old preprocessing with enhanced version for augmentation
    x = EnhancedPreprocessing()(x)
    
    # block 1
    x = layers.Conv1D(48, 3, strides=1, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(0.0005))(x)  # Light L2
    x = layers.BatchNormalization()(x)  # Add BN but keep MaxPooling same
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # block 2
    x = layers.Conv1D(96, 3, strides=1, padding="same", activation="relu",
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # block 3
    x = layers.Conv1D(192, 3, strides=1, padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)  # Slightly reduced from 0.5

    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    gesture_probs = layers.Dense(len(Config.gesture_labels),
                               activation="softmax", name="gesture_probs")(x)
    output = layers.Concatenate()([has_motion, gesture_probs])

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