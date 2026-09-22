# envisionhgdetector/detector.py

import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import cv2
import shutil
import time
from .config import Config
from .model_cnn import GestureModel  # Renamed CNN model
from .model_cnn_b import GestureModel as BinaryGestureModel  # New binary CNN model
from .model_lightgbm import LightGBMGestureModel  # New LightGBM model
from .model_combined import CombinedGestureModel, CombinedConfig  # Combined model
from .preprocessing import VideoProcessor, create_sliding_windows
from .utils import (
    create_segments, get_prediction_at_threshold, create_elan_file, 
    label_video, cut_video_by_segments, retrack_gesture_videos,
    compute_gesture_kinematics_dtw, create_gesture_visualization, create_dashboard,
    setup_dashboard_folders, joint_map, calc_mcneillian_space, calc_vert_height,
    calc_volume_size, calc_holds, get_label_from_prediction
)
from .label_video_combined import label_video_combined  # Dual-panel for combined model
from .default_config import DefaultConfig

# Standard library imports
import json
from pathlib import Path
import mediapipe as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import gaussian_filter1d
import umap.umap_ as umap
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from scipy import signal
from scipy.spatial.distance import euclidean
from typing import NamedTuple, Literal, get_args
from dataclasses import dataclass
import statistics
from .state import Thresholds, Row

# suppress warnings
import logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

def apply_smoothing(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply simple moving average smoothing to a series."""
    return series.rolling(window=window, center=True).mean().fillna(series)

VALID_MODEL_NAME_LITERAL = Literal["cnn", "cnn_b", "lightgbm", "combined"]
VALID_MODEL_NAMES = get_args(VALID_MODEL_NAME_LITERAL)

class GestureDetector:
    """Main class for gesture detection in videos - supports CNN, LightGBM, and Combined models."""
    
    def __init__(
        self,
        model_type: VALID_MODEL_NAME_LITERAL,
        config_path: Optional[Path] = None,
        weights_path: Optional[Path] = None,
        thresholds: Optional[Thresholds] = None
    ):
        """
        Initialize detector with model type selection.
        
        Args:
            model_type: "cnn", "cnn_b", "lightgbm"
            config_path: Optional path to model config file
            weights_path: Optional path to model weights file
            thresholds: Optional Thresholds object for model thresholds
        """
        if thresholds is None:
            thresholds = Thresholds()  # Use default thresholds if none provided
        self.thresholds = thresholds

        # Validate model type
        self.model_type = model_type
        if self.model_type not in VALID_MODEL_NAMES:
            raise ValueError(f"Unknown model type: {model_type}. Use one of {VALID_MODEL_NAMES}.")
        
        if self.model_type == "lightgbm":
            self.config = DefaultConfig("lightgbm", self.thresholds, config_path, weights_path).get_config()
            self.model = LightGBMGestureModel(self.config)
            self.video_processor = None  # LightGBM handles its own processing
            print(f"Initialized LightGBM gesture detector")

        elif self.model_type == "cnn_b":
            self.config = DefaultConfig("cnn_b", self.thresholds, config_path, weights_path).get_config()
            self.model = BinaryGestureModel(self.config)
            self.video_processor = VideoProcessor(self.config.seq_length)
            self.target_fps = self.config.target_fps or 25  # Default to 25 if not specified
            print(f"Initialized CNN-B gesture detector")

        else:  # CNN
            raise NotImplementedError("The 'cnn' model is not implemented yet. Please use 'cnn_b' or 'lightgbm'.")
            # self.model = GestureModel(self.config)
            # self.video_processor = VideoProcessor(self.config.seq_length)
            # print(f"Initialized CNN gesture detector")
                
    def set_thresholds(
        self,
        cnn_motion_threshold: Optional[float] = None,
        cnn_gesture_threshold: Optional[float] = None,
        lgbm_threshold: Optional[float] = None
    ):
        raise NotImplementedError("Threshold setting is not implemented yet. Please use the DefaultConfig class to set thresholds when initializing the GestureDetector.")
        """Update thresholds (combined model only)."""
        if self.model_type == "combined":
            self.model.set_thresholds(
                cnn_motion_threshold=cnn_motion_threshold,
                cnn_gesture_threshold=cnn_gesture_threshold,
                lgbm_threshold=lgbm_threshold
            )
        else:
            print(f"Warning: set_thresholds only applies to combined model")
    
    def _create_windows(self, features: List[List[float]], seq_length: int, stride: int) -> np.ndarray:
        """Creates sliding windows from feature sequences (CNN only)."""
        windows = []
        if len(features) < seq_length:
            return np.array([])
        for i in range(0, len(features) - seq_length + 1, stride):
            windows.append(features[i:i + seq_length])
        return np.array(windows)

    def _get_video_fps(self, video_path: str) -> int:
        """Get video FPS."""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def _expand_predictions_to_frames(
        self,
        predictions: pd.DataFrame,
        total_frames: int,
        fps: float
    ) -> pd.DataFrame:
        """Return one dataframe row for every source video frame.
        filled unavailable data with NoGesture and prediction_available=False
        """
        frame_df = pd.DataFrame({
            'frame_idx': np.arange(total_frames, dtype=np.int64),
        })
        frame_df['time'] = frame_df['frame_idx'] / fps if fps > 0 else np.nan

        if predictions.empty:
            frame_df['prediction'] = 'NoGesture'
            frame_df['prediction_available'] = False
            return frame_df

        dense_df = frame_df.merge(predictions, on=['frame_idx', 'time'], how='left')
        dense_df['prediction_available'] = dense_df['prediction'].notna()
        dense_df['prediction'] = dense_df['prediction'].fillna('NoGesture')
        return dense_df
    
    def predict_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray, List[float]]:
        """
        Process single video and return predictions.
        Automatically routes to appropriate model implementation.
        
        Returns:
            Tuple of (predictions_df, stats, segments_df, features_array, timestamps)
        """
        if self.model_type == "combined":
            return self._predict_video_combined(video_path, stride)
        elif self.model_type == "lightgbm":
            return self._predict_video_lightgbm(video_path, stride)
        elif self.model_type == "cnn_b":
            return self._predict_video_cnn_b(video_path, stride)
        else:
            return self._predict_video_cnn(video_path, stride)
    
    def _predict_video_combined(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray, List[float]]:
        """
        Combined CNN+LightGBM prediction - outputs BOTH models' results separately.
        No ensemble - user can compare both models.
        
        Returns:
            Tuple of (raw_df, stats, segments_df, raw_predictions, timestamps)
        """
        # Get predictions from combined model
        results = self.model.predict_video(video_path, target_fps=int(self.target_fps), return_all=True)
        
        if results['processed_frames'] == 0:
            return pd.DataFrame(), {"error": "No features detected"}, pd.DataFrame(), np.array([]), []
        
        fps = results['original_fps']
        rows = []
        timestamps = []
        
        for pred in results['predictions']:
            cnn = pred.get('cnn', {})
            lgbm = pred.get('lightgbm', {})
            time_s = pred['time_s']
            timestamps.append(time_s)
            
            # Build row with BOTH models' results
            row = {
                'time': time_s,
            }
            
            # Add CNN results
            if cnn:
                row.update({
                    'has_motion': cnn.get('has_motion', 0.0),
                    'Gesture_confidence': cnn.get('gesture_prob', 0.0) * cnn.get('has_motion', 1.0),
                    'Move_confidence': cnn.get('move_prob', 0.0) * cnn.get('has_motion', 1.0),
                    'NoGesture_confidence': 1 - cnn.get('has_motion', 0.0),
                    'cnn_class': cnn.get('class', 'NoGesture'),
                    'cnn_confidence': cnn.get('confidence', 0.0),
                })
            
            # Add LightGBM results
            if lgbm:
                row.update({
                    'lgbm_class': lgbm.get('class', 'NoGesture'),
                    'lgbm_confidence': lgbm.get('confidence', 0.0),
                    'lgbm_nogesture_prob': lgbm.get('nogesture_prob', 0.0),
                    'lgbm_gesture_prob': lgbm.get('gesture_prob', 0.0),
                })
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame(), {"error": "No predictions generated"}, pd.DataFrame(), np.array([]), []
        
        raw_df = pd.DataFrame(rows)
        
        # Create segments for BOTH models separately
        cnn_segments = self._create_segments_from_predictions(
            raw_df, 'cnn_class', 
            self.model.config.thresholds.cnn_motion_threshold
        ) if 'cnn_class' in raw_df.columns else pd.DataFrame()
        
        lgbm_segments = self._create_segments_from_predictions(
            raw_df, 'lgbm_class',
            self.model.config.thresholds.lgbm_threshold
        ) if 'lgbm_class' in raw_df.columns else pd.DataFrame()
        
        # Add model source to segments
        if not cnn_segments.empty:
            cnn_segments['model'] = 'CNN'
        if not lgbm_segments.empty:
            lgbm_segments['model'] = 'LightGBM'
        
        # Combine segments (user can filter by 'model' column)
        segments_df = pd.concat([cnn_segments, lgbm_segments], ignore_index=True)
        
        # Stats
        stats = {
            'total_frames': results['total_frames'],
            'processed_frames': results['processed_frames'],
            'fps': fps,
            'duration': results['total_frames'] / fps if fps > 0 else 0,
            'cnn_available': results['cnn_available'],
            'lgbm_available': results['lgbm_available'],
            'cnn_motion_threshold': self.model.config.thresholds.cnn_motion_threshold,
            'cnn_gesture_threshold': self.model.config.thresholds.cnn_gesture_threshold,
            'lgbm_threshold': self.model.config.thresholds.lgbm_threshold,
        }
        
        # Raw predictions array for compatibility
        if 'has_motion' in raw_df.columns:
            raw_predictions = raw_df[['has_motion', 'Gesture_confidence', 'Move_confidence']].values
        else:
            raw_predictions = np.array([])
        
        return raw_df, stats, segments_df, raw_predictions, timestamps
    
    def _create_segments_from_predictions(
        self, 
        raw_df: pd.DataFrame, 
        class_column: str,
        threshold: float
    ) -> pd.DataFrame:
        """Create segments from a specific model's predictions."""
        if raw_df.empty or class_column not in raw_df.columns:
            return pd.DataFrame(columns=['start_time', 'end_time', 'prediction', 'prediction_id', 'duration'])
        
        segments = []
        segment_id = 1
        in_gesture = False
        start_idx = 0
        
        min_length_s = self.config.thresholds.min_length_s
        min_gap_s = self.config.thresholds.min_gap_s
        
        for idx, row in raw_df.iterrows():
            is_gesture = row[class_column] != 'NoGesture'
            
            if is_gesture and not in_gesture:
                in_gesture = True
                start_idx = idx
            elif not is_gesture and in_gesture:
                in_gesture = False
                start_time = raw_df.loc[start_idx, 'time']
                end_time = raw_df.loc[idx - 1, 'time'] if idx > 0 else raw_df.loc[idx, 'time']
                duration = end_time - start_time
                
                if duration >= min_length_s:
                    # Get majority label in segment
                    segment_data = raw_df.loc[start_idx:idx-1]
                    label = segment_data[class_column].mode().iloc[0] if not segment_data[class_column].mode().empty else 'Gesture'
                    
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'prediction': label,
                        'prediction_id': segment_id,
                        'duration': duration
                    })
                    segment_id += 1
        
        # Handle gesture at end
        if in_gesture:
            start_time = raw_df.loc[start_idx, 'time']
            end_time = raw_df.iloc[-1]['time']
            duration = end_time - start_time
            
            if duration >= min_length_s:
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'prediction': 'Gesture',
                    'prediction_id': segment_id,
                    'duration': duration
                })
        
        # Merge close segments
        if len(segments) > 1:
            merged = []
            current = segments[0]
            
            for next_seg in segments[1:]:
                gap = next_seg['start_time'] - current['end_time']
                if gap <= min_gap_s:
                    current['end_time'] = next_seg['end_time']
                    current['duration'] = current['end_time'] - current['start_time']
                else:
                    merged.append(current)
                    current = next_seg
            merged.append(current)
            segments = merged
        
        return pd.DataFrame(segments) if segments else pd.DataFrame(columns=['start_time', 'end_time', 'prediction', 'prediction_id', 'duration'])
    
    def _predict_video_cnn(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """Original CNN prediction method."""
        # Extract features and timestamps
        features, timestamps, frame_indices = self.video_processor.process_video(video_path)
    
        if not features:
            return pd.DataFrame(), {"error": "No features detected"}, pd.DataFrame(), np.array([])
        
        windows = self._create_windows(features, self.config.seq_length, stride)
        
        if len(windows) == 0:
            return pd.DataFrame(), {"error": "No valid windows created"}, pd.DataFrame(), np.array([])

        # Get predictions
        predictions = self.model.predict(windows)
        
        # Create results DataFrame - use the actual timestamps for frames with valid skeleton data
        fps = self._get_video_fps(video_path)
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

                rows.append({
                    'time': time+((self.config.seq_length / 2) / self.target_fps),
                    'has_motion': float(has_motion),
                    'NoGesture_confidence': float(1 - has_motion),
                    'Gesture_confidence': gesture_confidence,
                    'Move_confidence': move_confidence
                })
            else:
                rows.append({
                    'time': time+((self.config.seq_length / 2) / self.target_fps),
                    'has_motion': float(has_motion),
                    'NoGesture_confidence': float(1 - has_motion),
                    'Gesture_confidence': float(gesture_probs[0]),
                    'Move_confidence': float(gesture_probs[1])
                })
        
        results_df = pd.DataFrame(rows)

        # Apply thresholds
        results_df['prediction'] = results_df.apply(
            lambda row: get_prediction_at_threshold(
                row,
                self.model.config.thresholds.motion_threshold,
                self.model.config.thresholds.gesture_threshold
            ),
            axis=1
        )

        # Create segments
        segments = create_segments(
            results_df,
            label_column='prediction',
            min_gap_s=self.model.config.thresholds.min_gap_s,
            min_length_s=self.model.config.thresholds.min_length_s
        )

        # Calculate statistics
        stats = {
            'average_motion': float(results_df['has_motion'].mean()),
            'average_gesture': float(results_df['Gesture_confidence'].mean()),
            'average_move': float(results_df['Move_confidence'].mean()),
            'applied_gesture_class_bias': float(gesture_class_bias),
            'model_type': self.model_type
        }
        
        return results_df, stats, segments, features, timestamps

    def _predict_video_cnn_b(
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

        if not features:
            return pd.DataFrame(), {"error": "No features detected"}, pd.DataFrame(), np.array([]), []

        results_df = self._predict_video_cnn_b_from_features(features, stride)

        segments = create_segments(
            results_df,
            label_column='prediction',
            min_gap_s=self.model.config.thresholds.min_gap_s,
            min_length_s=self.model.config.thresholds.min_length_s
        )

        stats = {
            'average_gesture': float(results_df['confidence'].mean()),
            'model_type': self.model_type
        }

        return results_df, stats, segments, features, timestamps
    
    def _predict_video_cnn_from_features(
        self,
        features: np.ndarray,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """CNN prediction from landmarks."""
        windows = self._create_windows(features, self.config.seq_length, stride)
        
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

                rows.append({
                    'has_motion': float(has_motion),
                    'NoGesture_confidence': float(1 - has_motion),
                    'Gesture_confidence': gesture_confidence,
                    'Move_confidence': move_confidence
                })
            else:
                rows.append({
                    'has_motion': float(has_motion),
                    'NoGesture_confidence': float(1 - has_motion),
                    'Gesture_confidence': float(gesture_probs[0]),
                    'Move_confidence': float(gesture_probs[1])
                })
        
        results_df = pd.DataFrame(rows)

        # Apply thresholds
        results_df['prediction'] = results_df.apply(
            lambda row: get_prediction_at_threshold(
                row,
                self.model.config.thresholds.motion_threshold,
                self.model.config.thresholds.gesture_threshold
            ),
            axis=1
        )
        return results_df

    def _predict_video_cnn_b_from_features(self, features: np.ndarray, stride: int = 1) -> pd.DataFrame:
        # TODO
        '''
        Behaviorally, this is per-feature-frame majority voting, not per-original-video-frame prediction. Frames that are not represented in features cannot be recovered by this method alone.
        '''
        """Predict one binary label for every extracted feature frame."""
        windows = self._create_windows(features, self.config.seq_length, stride)
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

            label = get_label_from_prediction(
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
                frame_predictions.append("NoGesture")
                continue

            gesture_votes = votes.count("Gesture")
            no_gesture_votes = votes.count("NoGesture")
            frame_predictions.append(
                "Gesture" if gesture_votes > no_gesture_votes else "NoGesture"
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
                no_gesture_confidence= 1.0 - frame_probabilities[frame_idx]
            ))

        results = pd.DataFrame([vars(row) for row in all_predictions])

        return results
    
    def _predict_video_lightgbm(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """LightGBM prediction method with CNN-compatible output."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video with LightGBM: {fps:.1f}fps, {total_frames} frames")
        
        # Reset model state
        self.model.key_joints_buffer.clear()
        if hasattr(self.model, 'left_fingers_buffer'):
            self.model.left_fingers_buffer.clear()
            self.model.right_fingers_buffer.clear()
        
        predictions = []
        frame_number = 0
        valid_features = []  # Store extracted features for compatibility
        valid_timestamps = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames based on stride
            if frame_number % stride != 0:
                frame_number += 1
                continue
            
            timestamp = frame_number / fps
            
            # Extract features using LightGBM model
            features = self.model.extract_features_from_frame(frame)
            
            if features is not None:
                # Store for compatibility
                valid_features.append(features.tolist())
                valid_timestamps.append(timestamp)
                
                # Get prediction
                pred_probs = self.model.predict(features.reshape(1, -1))[0]
                predicted_class = np.argmax(pred_probs)
                confidence = pred_probs[predicted_class]
                
                # Convert to gesture name
                gesture_name = self.model.label_encoder.inverse_transform([predicted_class])[0]
                gesture_name = self.model.standardize_gesture_name(gesture_name)
                
                # Convert LightGBM output to align witht he CNN format
                if gesture_name == "NOGESTURE":
                    gesture_conf = 1-confidence # gesture confidence is the 1-no gesture confidence                   
                    nogesture_conf = confidence # no gesture confidence is the confidence of the no gesture class
                    move_conf = 0.0
                else:
                    # Distribute confidence based on gesture type
                    if "move" in gesture_name.lower() or "MOVE" in gesture_name:
                        gesture_conf = 0.0
                        move_conf = confidence
                        nogesture_conf = 1-confidence
                    else: #then its a a gesture
                        gesture_conf = confidence
                        move_conf = 0.0
                        nogesture_conf = 1-confidence
                
                predictions.append({
                    'frame_idx': frame_number,
                    'time': timestamp,
                    'has_motion': gesture_conf,
                    'NoGesture_confidence': nogesture_conf,
                    'Gesture_confidence': gesture_conf,
                    'Move_confidence': move_conf
                })
            
            frame_number += 1
            
            # Progress update
            if frame_number % 500 == 0:
                progress = frame_number / total_frames * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        
        # Convert to DataFrame
        sparse_results_df = pd.DataFrame(predictions)
        
        if sparse_results_df.empty:
            return pd.DataFrame(), {"error": "No predictions generated"}, pd.DataFrame(), np.array([])
        
        # Apply thresholds (reuse existing logic)
        sparse_results_df['prediction'] = sparse_results_df.apply(
            lambda row: get_prediction_at_threshold(
                row,
                self.model.config.thresholds.motion_threshold,
                self.model.config.thresholds.gesture_threshold
            ),
            axis=1
        )

        # Create segments
        segments = create_segments(
            sparse_results_df,
            label_column='prediction',
            min_gap_s=self.model.config.thresholds.min_gap_s,
            min_length_s=self.model.config.thresholds.min_length_s
        )

        # Calculate statistics
        stats = {
            'average_motion': float(sparse_results_df['has_motion'].mean()),
            'average_gesture': float(sparse_results_df['Gesture_confidence'].mean()),
            'average_move': float(sparse_results_df['Move_confidence'].mean()),
            'model_type': self.model_type,
            'lightgbm_features': len(valid_features)
        }
        
        results_df = self._expand_predictions_to_frames(
            sparse_results_df,
            total_frames,
            fps
        )
        return results_df, stats, segments, np.array(valid_features), valid_timestamps

    def _predict_video_lightgbm_from_features(
        self,
        landmarks_per_frame: np.ndarray,
        fps: float,
    ) -> pd.DataFrame:
        """LightGBM prediction from landmarks method with CNN-compatible output."""

        # Reset model state
        self.model.key_joints_buffer.clear()
        if hasattr(self.model, 'left_fingers_buffer'):
            self.model.left_fingers_buffer.clear()
            self.model.right_fingers_buffer.clear()
        
        predictions = []
        frame_number = 0

        for landmarks in landmarks_per_frame:
            timestamp = frame_number / fps
            features = self.model.extract_features_from_landmarks(landmarks)
            if features is not None:
                try:
                    pred_probs = self.model.predict(features.reshape(1, -1))[0]
                except Exception as e:
                    raise RuntimeError(f"Prediction failed for frame {frame_number} at timestamp {timestamp:.2f}s: \n{e}")
                
                predicted_class = np.argmax(pred_probs)
                confidence = pred_probs[predicted_class]
                
                # Convert to gesture name
                gesture_name = self.model.label_encoder.inverse_transform([predicted_class])[0]

                # Convert LightGBM output to align witht he CNN format
                if gesture_name == "NoGesture":
                    gesture_conf = 1-confidence # gesture confidence is the 1-no gesture confidence                   
                    nogesture_conf = confidence # no gesture confidence is the confidence of the no gesture class
                    move_conf = 0.0
                elif gesture_name == "Gesture":
                    gesture_conf = confidence
                    move_conf = 0.0
                    nogesture_conf = 1-confidence
                elif gesture_name == "Move": # TODO - do we need this?
                    gesture_conf = 0.0
                    move_conf = confidence
                    nogesture_conf = 1-confidence
                else:
                    raise ValueError(f"Unexpected gesture name '{gesture_name}' for frame {frame_number} at timestamp {timestamp:.2f}s")
                    
            else:
                nogesture_conf = 1.0
                gesture_conf = 0.0
                move_conf = 0.0

            prediction = get_label_from_prediction(
                nogesture_conf,
                gesture_conf,
                move_conf,
                self.model.config.thresholds.motion_threshold,
                self.model.config.thresholds.gesture_threshold
            )
            predictions.append(Row(
                frame_index=frame_number,
                prediction=prediction,
                confidence=gesture_conf,
                gesture_confidence=gesture_conf,
                no_gesture_confidence=nogesture_conf,
                move_confidence=move_conf,
                motion_confidence=gesture_conf,
                timestamp=timestamp
            ))
            
            frame_number += 1
            
            # Progress update
            if frame_number % 500 == 0:
                progress = frame_number / len(landmarks_per_frame) * 100
                print(f"Progress: {progress:.1f}%")
                
        # Convert to DataFrame
        results_df = pd.DataFrame(vars(prediction) for prediction in predictions)
        
        if results_df.empty:
            return pd.DataFrame(), {"error": "No predictions generated"}, pd.DataFrame(), np.array([])

        return results_df

    def predict_labels_from_landmarks(self, landmarks_per_frame: np.ndarray, fps: float) -> pd.DataFrame:
        if self.model_type == "lightgbm":
            results_df = self._predict_video_lightgbm_from_features(landmarks_per_frame, fps)
        elif self.model_type == "cnn":
            results_df = self._predict_video_cnn_from_features(landmarks_per_frame)
        elif self.model_type == "cnn_b":
            results_df = self._predict_video_cnn_b_from_features(landmarks_per_frame)

        return results_df

    def process_video(self, video_path: str, output_folder: str, elan_only: bool = False):
        output = dict()
        print("Elan only flag is set to:", elan_only)
        if not os.path.exists(video_path):
            output["error"] = f"Video not found: {video_path}"
            return output

        os.makedirs(output_folder, exist_ok=True)

        video_name = os.path.basename(video_path)
        print(f"\nProcessing {video_name} with {self.model_type.upper()} model...")
        
        try:
            # Process video (automatically routes to correct model)
            print("Extracting features and model inferencing...")
            predictions_df, stats, segments, features, timestamps = self.predict_video(video_path)
            
            if not predictions_df.empty:
                # Save predictions
                if not elan_only:
                    output_pathpred = os.path.join(
                        output_folder,
                        f"{video_name}_predictions.csv"
                    )
                    predictions_df.to_csv(output_pathpred, index=False)
                    
                    # Save segments
                    output_pathseg = os.path.join(
                        output_folder,
                        f"{video_name}_segments.csv"
                    )
                    segments.to_csv(output_pathseg, index=False)

                    # Save features (if available)
                    if len(features) > 0:
                        output_pathfeat = os.path.join(
                            output_folder,
                            f"{video_name}_features.npy"
                        )
                        feature_array = np.array(features)
                        np.save(output_pathfeat, feature_array)

                # Labeled video generation
                    print("Generating labeled video...")
                    output_pathvid = os.path.join(
                        output_folder,
                        f"labeled_{video_name}"
                    )
                    label_video(
                        video_path, 
                        segments, 
                        output_pathvid,
                        predictions_df,
                        valid_timestamps=timestamps,
                        motion_threshold=self.model.config.thresholds.motion_threshold,
                        gesture_threshold=self.model.config.thresholds.gesture_threshold,
                        target_fps=25.0
                    )
                
                print("Generating ELAN file...")
                # Create ELAN file
                output_path = os.path.join(
                    output_folder,
                    f"{video_name}.eaf"
                )
                fps = self._get_video_fps(video_path)
                create_elan_file(
                    video_path,
                    segments,
                    output_path,
                    fps=fps,
                    include_ground_truth=False
                )

                output['stats'] = stats
                output['output_path'] = output_path
                print(f"Done processing {video_name} with {self.model_type.upper()}")
            else:
                output["error"] = "No predictions generated"

        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            output["error"] =  str(e)
        
        return output
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        video_pattern: str = "*.mp4"
    ) -> Dict[str, Dict]:
        """Process all videos in a folder (works with both CNN and LightGBM)."""
        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all videos
        videos = glob.glob(os.path.join(input_folder, video_pattern))
        results = {}
        
        for video_path in videos:
            video_name = os.path.basename(video_path)
            results[video_name] = self.process_video(video_path, output_folder)
            
        return results
        
    def retrack_gestures(
        self,
        input_folder: str,
        output_folder: str
    ) -> Dict[str, str]:
        """Retrack gesture segments using MediaPipe world landmarks (works with both models)."""
        try:
            # Retrack the videos and save landmarks
            tracked_data = retrack_gesture_videos(
                input_folder=input_folder,
                output_folder=output_folder
            )
            
            if not tracked_data:
                return {"error": "No gestures could be tracked"}
                
            print(f"Successfully retracked {len(tracked_data)} gestures")
            
            return {
                "tracked_folder": os.path.join(output_folder, "tracked_videos"),
                "landmarks_folder": output_folder
            }
            
        except Exception as e:
            print(f"Error during gesture retracking: {str(e)}")
            return {"error": str(e)}

    def analyze_dtw_kinematics(
        self,
        landmarks_folder: str,
        output_folder: str,
        fps: float = 25.0
    ) -> Dict[str, str]:
        """Compute DTW distances, kinematic features, and create visualization (works with both models)."""
        try:
            # Compute DTW distances and kinematic features
            print("Computing DTW distances and kinematic features...")
            dtw_matrix, gesture_names, kinematic_features = compute_gesture_kinematics_dtw(
                tracked_folder=landmarks_folder,
                output_folder=output_folder,
                fps=fps
            )
            
            # Create visualization
            print("Creating visualization...")
            create_gesture_visualization(
                dtw_matrix=dtw_matrix,
                gesture_names=gesture_names,
                output_folder=output_folder
            )
            
            return {
                "distance_matrix": os.path.join(output_folder, "dtw_distances.csv"),
                "kinematic_features": os.path.join(output_folder, "kinematic_features.csv"),
                "visualization": os.path.join(output_folder, "gesture_visualization.csv")
            }
            
        except Exception as e:
            print(f"Error during DTW and kinematic analysis: {str(e)}")
            return {"error": str(e)}
    
    def prepare_gesture_dashboard(self, data_folder: str, assets_folder: Optional[str] = None) -> None:
        """Prepare dashboard (works with both models)."""
        try:
            if assets_folder is None:
                assets_folder = os.path.join(os.path.dirname(data_folder), "assets")

            # Set up folders and copy necessary files
            setup_dashboard_folders(data_folder, assets_folder)
            
            # Get the output directory (parent of analysis folder)
            output_dir = os.path.dirname(data_folder)
            
            # Copy the app.py to the output directory
            dashboard_script_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
            destination_script_path = os.path.join(output_dir, "app.py")
            shutil.copy(dashboard_script_path, destination_script_path)
            
            print(f"Dashboard prepared for {self.model_type.upper()} results")
            print(f"App dashboard copied to: {destination_script_path}")
            
            # Create the CSS file in the assets folder
            css_content = '''
                body, 
                .dash-graph,
                .dash-core-components,
                .dash-html-components { 
                    margin: 0; 
                    background-color: #111; 
                    font-family: sans-serif !important;
                    min-height: 100vh;
                    width: 100%;
                    color: #ffffff;
                }

                /* Modern container styling */
                .dashboard-container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 2rem;
                    font-family: sans-serif !important;
                }

                /* Enhanced headings */
                h1, h2, h3, h4, h5, h6 {
                    color: rgba(255, 255, 255, 0.95);
                    font-weight: 600;
                    letter-spacing: -0.02em;
                    font-family: sans-serif !important;
                }

                h1 {
                    font-size: 2.5rem;
                    text-align: center;
                    margin-bottom: 2rem;
                    background: linear-gradient(45deg, #fff, #a8a8a8);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    text-shadow: 0 0 30px rgba(255,255,255,0.1);
                    font-family: sans-serif !important;
                }

                h2 {
                    font-size: 1.5rem;
                    margin: 1.5rem 0;
                    padding-bottom: 0.5rem;
                    border-bottom: 2px solid rgba(255,255,255,0.1);
                    font-family: sans-serif !important;
                }

                /* Card-like sections */
                .visualization-section {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    backdrop-filter: blur(10px);
                }

                /* Grid layout for kinematic features */
                .kinematic-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1.5rem;
                    margin-right: 120px; /* Space for fixed video */
                    grid-auto-rows: minmax(200px, auto); 
                    height: 500px; /* Adjust as needed */
                }

                /* Video container styling */
                .video-container {
                    background: rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    padding: 1rem;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                }

                /* Interactive elements */
                .interactive-element {
                    transition: all 0.2s ease-in-out;
                }

                .interactive-element:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
                }

                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }

                ::-webkit-scrollbar-track {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                }

                ::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.3);
                    border-radius: 4px;
                }

                ::-webkit-scrollbar-thumb:hover {
                    background: rgba(255, 255, 255, 0.4);
                }

                /* Loading states */
                .loading {
                    opacity: 0.7;
                    transition: opacity 0.3s ease;
                }

                /* Tooltip styling */
                .tooltip {
                    background: rgba(0, 0, 0, 0.8);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    padding: 0.5rem;
                    font-size: 0.875rem;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                    font-family: sans-serif !important;
                }

                /* Force Dash components to use sans-serif */
                .dash-plot-container, 
                .dash-graph-container,
                .js-plotly-plot,
                .plotly {
                    font-family: sans-serif !important;
                }
                '''
            css_file_path = os.path.join(assets_folder, "styles.css")
            with open(css_file_path, "w") as css_file:
                css_file.write(css_content.strip())
            
            print(f"CSS file created at: {css_file_path}")
            print("Run 'python app.py' to start the dashboard")
            
        except Exception as e:
            print(f"Error preparing dashboard: {str(e)}")
            raise

