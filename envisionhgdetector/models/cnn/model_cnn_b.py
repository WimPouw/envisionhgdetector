# envisionhgdetector/envisionhgdetector/model.py
"""
Gesture detection CNN model.
Architecture matches best performing config from hyperparameter search.

Best model: World landmarks (92 features) with residual CNN blocks.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from tensorflow.keras import layers, regularizers

from envisionhgdetector.utils import get_prediction, create_segments, get_video_fps 
from envisionhgdetector.state import PredictionColumns, Row, Labels, CNN_B_Config, ModelNames
from .cnn_utils import make_model, create_windows
from .preprocessing import VideoProcessor
from ..model_template import ModelTemplate

class GestureModel(ModelTemplate):
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
        print('THIS IS NEW CNN B CODE')
        self.config = config
        self.model = make_model(config=config, type="binary")
        self.video_processor = VideoProcessor(seq_length=config.seq_length, feature_set=config.dataset_name)
    
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
        classes, _ = self.predict_with_confidence(features, motion_threshold)
        return classes

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

    def predict_video_from_landmarks(self, features: np.ndarray, fps:float, stride: int = 1) -> pd.DataFrame:
        # TODO
        '''
        Behaviorally, this is per-feature-frame majority voting, not per-original-video-frame prediction. Frames that are not represented in features cannot be recovered by this method alone.
        '''
        """Predict one binary label for every extracted feature frame."""
        windows = create_windows(features, self.config.seq_length, stride)
        if len(windows) == 0:
            return pd.DataFrame()

        window_predictions = self.model.predict(windows).reshape(-1)

        frame_votes = [[] for _ in range(len(features))]
        frame_probability_sums = np.zeros(len(features), dtype=float)
        frame_vote_counts = np.zeros(len(features), dtype=np.int64)

        for window_index, gesture_probability in enumerate(window_predictions):
            start = window_index * stride
            end = start + self.config.seq_length
            gesture_probability = float(gesture_probability)
            no_gesture_probability = 1 - gesture_probability
            move_probability = 0.0  # CNN-B does not predict move probability

            label = get_prediction(
                no_gesture_probability,
                gesture_probability,
                move_probability,
                self.config.thresholds.motion_threshold,
                self.config.thresholds.gesture_threshold
            )

            for frame_index in range(start, min(end, len(features))):
                frame_votes[frame_index].append(label)
                frame_probability_sums[frame_index] += gesture_probability
                frame_vote_counts[frame_index] += 1

        frame_probabilities = np.zeros(len(features), dtype=float)
        valid_frames = frame_vote_counts > 0
        frame_probabilities[valid_frames] = (
            frame_probability_sums[valid_frames] / frame_vote_counts[valid_frames]
        )

        frame_predictions = []
        for votes in frame_votes:
            if not votes:
                frame_predictions.append(Labels.NOGESTURE)  # Default to NoGesture if no votes
                continue

            gesture_votes = votes.count(Labels.GESTURE)
            no_gesture_votes = votes.count(Labels.NOGESTURE)
            frame_predictions.append(
                Labels.GESTURE if gesture_votes > no_gesture_votes else Labels.NOGESTURE
            )

        all_predictions = []
        for frame_idx, prediction in enumerate(frame_predictions):
            all_predictions.append(Row(
                frame_index=frame_idx,
                prediction=prediction,
                confidence=frame_probabilities[frame_idx],
                gesture_confidence= frame_probabilities[frame_idx],
                motion_confidence= frame_probabilities[frame_idx],
                move_confidence = 0.0,  # CNN-B does not predict move confidence
                no_gesture_confidence=1.0 - frame_probabilities[frame_idx],
                timestamp=frame_idx / fps,
            ))

        results = pd.DataFrame([vars(row) for row in all_predictions])

        return results

    def predict_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray, List[float]]:
        """Predict video labels with the binary CNN-B model."""
        # TODO
        '''
        One additional issue: results_df currently uses frame_index, while the video path also has frame_indices. They are not necessarily the same thing. Later, you should map the feature-frame indices back to frame_indices before treating them as original video-frame indices.
        '''
        features, timestamps, frame_indices = self.video_processor.process_video(video_path)
        print(f"Extracted {len(features)} feature frames from video: {video_path}")

        if not features:
            return pd.DataFrame(), {"error": "No features detected"}, pd.DataFrame(), np.array([]), []

        fps = get_video_fps(video_path)
        results_df = self.predict_video_from_landmarks(features, fps, stride)
        if not results_df.empty:
            if len(timestamps) < len(results_df):
                raise ValueError(
                    "Video processor returned fewer timestamps than feature frames."
                )
            results_df[PredictionColumns.TIMESTAMP] = np.asarray(timestamps[:len(results_df)])

        print("Creating Segments")
        try:
            segments = create_segments(
                results_df,
                min_gap_s=self.config.thresholds.min_gap_s,
                min_length_s=self.config.thresholds.min_length_s
            )
        except Exception as e:
            print(f"Error creating segments: {e}")
            segments = pd.DataFrame()

        print(f"Created {len(segments)} segments from predictions.")

        print('Creating Statistics')
        stats = {
            'average_gesture': float(results_df[PredictionColumns.GESTURE_CONFIDENCE].mean()),
            'model_type': ModelNames.CNN_B,
        }

        return results_df, stats, segments, features, timestamps
    
    
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