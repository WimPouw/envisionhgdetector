# envisionhgdetector/envisionhgdetector/model.py
"""
Gesture detection CNN model.
Architecture matches best performing config from hyperparameter search.

Best model: World landmarks (92 features) with residual CNN blocks.
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from typing import Optional, Tuple, Dict, List
import pandas as pd
import numpy as np
from ..utils import get_prediction_at_threshold, create_segments, get_video_fps
from .cnn_utils import make_model, create_windows
from ..state import Row, Labels, CNN_Config
from ..preprocessing import VideoProcessor

class GestureModel:
    """
    Wrapper class for the gesture detection model.
    Handles model loading and inference.
    """
    
    def __init__(self, config: CNN_Config):
        """
        Initialize the model.
        
        Args:
            config: The configuration object for the model.
        """
        self.config = config
        self.model = make_model(config=config, type="multi")
        self.video_processor = VideoProcessor(seq_length=config.seq_length, target_fps=config.target_fps)
    
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
    
    def predict_classes(self, features: np.ndarray, motion_threshold: float) -> np.ndarray:
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
        motion_threshold: float
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

    def _predict_video_from_landmarks(
        self,
        features: np.ndarray,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """CNN prediction from landmarks."""
        windows = create_windows(features, self.config.seq_length, stride)
        
        if len(windows) == 0:
            return pd.DataFrame(), {"error": "No valid windows created"}, pd.DataFrame(), np.array([])

        # Get predictions
        predictions = self.model.predict(windows)
        
        # Create results DataFrame - use the actual timestamps for frames with valid skeleton data
        rows = []
        gesture_class_bias = self.config.thresholds.gesture_class_bias
        
        for i, (pred) in enumerate(predictions):
            has_motion = pred[0]
            gesture_probs = pred[1:]
            
            # Apply bias if configured
            if gesture_class_bias is not None and abs(gesture_class_bias) >= 1e-9:
                gesture_confidence = float(gesture_probs[0])
                move_confidence = float(gesture_probs[1])
                
                if has_motion > 0:
                    total_conf = gesture_confidence + move_confidence
                    if total_conf > 0:
                        adjustment = gesture_class_bias * move_confidence * 0.5
                        adjusted_gesture = gesture_confidence + adjustment
                        adjusted_move = move_confidence - adjustment
                        
                        if adjusted_gesture + adjusted_move > 0:
                            norm_factor = total_conf / (adjusted_gesture + adjusted_move)
                            adjusted_gesture *= norm_factor
                            adjusted_move *= norm_factor
                        
                        gesture_confidence = adjusted_gesture
                        move_confidence = adjusted_move

                prediction = get_prediction_at_threshold(
                    has_motion,
                    gesture_confidence,
                    move_confidence,
                    self.config.thresholds.motion_threshold,
                    self.config.thresholds.gesture_threshold
                )
                rows.append(
                    Row(
                        frame_index=i,
                        prediction=prediction,
                        confidence=gesture_confidence if has_motion else 1 - has_motion,
                        motion_confidence=has_motion,
                        gesture_confidence=gesture_confidence,
                        move_confidence=move_confidence,
                        no_gesture_confidence=1 - has_motion,
                        timestamp=None  # Timestamp can be added if available
                    )
                )
            else:
                prediction = get_prediction_at_threshold(
                    has_motion,
                    gesture_probs[0],
                    gesture_probs[1],
                    self.config.thresholds.motion_threshold,
                    self.config.thresholds.gesture_threshold
                )
                rows.append(
                    Row(
                        frame_index=i,
                        prediction=prediction,
                        confidence=gesture_probs[0] if has_motion else 1 - has_motion,
                        motion_confidence=has_motion,
                        gesture_confidence=gesture_probs[0],
                        move_confidence=gesture_probs[1],
                        no_gesture_confidence=1 - has_motion,
                        timestamp=None  # Timestamp can be added if available
                    )
                )
        
        results_df = pd.DataFrame([vars(row) for row in rows])
        return results_df

    def _predict_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """Original CNN prediction method."""
        # Extract features and timestamps
        features, timestamps, frame_indices = self.video_processor.process_video(video_path)
    
        if not features:
            return pd.DataFrame(), {"error": "No features detected"}, pd.DataFrame(), np.array([])
        
        windows = create_windows(features, self.config.seq_length, stride)
        
        if len(windows) == 0:
            return pd.DataFrame(), {"error": "No valid windows created"}, pd.DataFrame(), np.array([])

        # Get predictions
        predictions = self.model.predict(windows)
        
        # Create results DataFrame - use the actual timestamps for frames with valid skeleton data
        fps = get_video_fps(video_path)
        rows = []
        gesture_class_bias = self.config.thresholds.gesture_class_bias
        
        for i, (pred, time) in enumerate(zip(predictions, timestamps[::stride])):
            has_motion = pred[0]
            gesture_probs = pred[1:]
            
            # Apply bias if configured
            if gesture_class_bias is not None and abs(gesture_class_bias) >= 1e-9:
                gesture_confidence = float(gesture_probs[0])
                move_confidence = float(gesture_probs[1])
                
                if has_motion > 0:
                    total_conf = gesture_confidence + move_confidence
                    if total_conf > 0:
                        adjustment = gesture_class_bias * move_confidence * 0.5
                        adjusted_gesture = gesture_confidence + adjustment
                        adjusted_move = move_confidence - adjustment
                        
                        if adjusted_gesture + adjusted_move > 0:
                            norm_factor = total_conf / (adjusted_gesture + adjusted_move)
                            adjusted_gesture *= norm_factor
                            adjusted_move *= norm_factor
                        
                        gesture_confidence = adjusted_gesture
                        move_confidence = adjusted_move

                prediction = get_prediction_at_threshold(
                    has_motion,
                    gesture_confidence,
                    move_confidence,
                    self.config.thresholds.motion_threshold,
                    self.config.thresholds.gesture_threshold
                )
                rows.append(Row(
                    frame_index=i,
                    prediction=prediction,
                    timestamp=time+((self.config.seq_length / 2) / self.config.target_fps),
                    confidence=gesture_confidence if has_motion else 1 - has_motion,
                    motion_confidence=has_motion,
                    gesture_confidence=gesture_confidence,
                    move_confidence=move_confidence,
                    no_gesture_confidence=1 - has_motion
                    ))
            else:
                prediction = get_prediction_at_threshold(
                    has_motion,
                    gesture_probs[0],
                    gesture_probs[1],
                    self.config.thresholds.motion_threshold,
                    self.config.thresholds.gesture_threshold
                )
                rows.append(Row(
                    frame_index=i,
                    prediction=prediction,
                    timestamp=time+((self.config.seq_length / 2) / self.config.target_fps),
                    confidence=gesture_probs[0] if has_motion else 1 - has_motion,
                    motion_confidence=has_motion,
                    gesture_confidence=gesture_probs[0],
                    move_confidence=gesture_probs[1],
                    no_gesture_confidence=1 - has_motion
                ))
        
        results_df = pd.DataFrame(vars(row) for row in rows)

        # Create segments
        segments = create_segments(
            results_df,
            label_column='prediction',
            min_gap_s=self.model.config.thresholds.min_gap_s,
            min_length_s=self.model.config.thresholds.min_length_s
        )

        # Calculate statistics
        stats = {
            'average_motion': float(results_df['motion_confidence'].mean()),
            'average_gesture': float(results_df['gesture_confidence'].mean()),
            'average_move': float(results_df['move_confidence'].mean()),
            'applied_gesture_class_bias': float(gesture_class_bias),
            'model_type': self.model_type
        }
        
        return results_df, stats, segments, features, timestamps

    

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