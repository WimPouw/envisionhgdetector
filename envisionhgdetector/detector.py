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
from .model_lightgbm import LightGBMGestureModel  # New LightGBM model
from .preprocessing import VideoProcessor, create_sliding_windows
from .utils import (
    create_segments, get_prediction_at_threshold, create_elan_file, 
    label_video, cut_video_by_segments, retrack_gesture_videos,
    compute_gesture_kinematics_dtw, create_gesture_visualization, create_dashboard,
    setup_dashboard_folders, joint_map, calc_mcneillian_space, calc_vert_height,
    calc_volume_size, calc_holds
)

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
from typing import NamedTuple
from dataclasses import dataclass
import statistics

# suppress warnings
import logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

def apply_smoothing(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply simple moving average smoothing to a series."""
    return series.rolling(window=window, center=True).mean().fillna(series)
    

class GestureDetector:
    """Main class for gesture detection in videos - supports both CNN and LightGBM."""
    
    def __init__(
        self,
        model_type: str = "cnn",  # NEW: "cnn" or "lightgbm"
        motion_threshold: Optional[float] = None,
        gesture_threshold: Optional[float] = None,
        min_gap_s: Optional[float] = None,
        min_length_s: Optional[float] = None,
        gesture_class_bias: float = 0.0,
        config: Optional[Config] = None
    ):
        """Initialize detector with model type selection."""
        self.config = config or Config()
        self.model_type = model_type.lower()
        
        # Validate model type
        if self.model_type not in ["cnn", "lightgbm"]:
            raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'lightgbm'")
        
        self.params = {
            'motion_threshold': motion_threshold or self.config.default_motion_threshold,
            'gesture_threshold': gesture_threshold or self.config.default_gesture_threshold,
            'min_gap_s': min_gap_s or self.config.default_min_gap_s,
            'min_length_s': min_length_s or self.config.default_min_length_s,
            'gesture_class_bias': gesture_class_bias
        }
        
        # Initialize model based on type
        if self.model_type == "lightgbm":
            self.model = LightGBMGestureModel(self.config)
            self.video_processor = None  # LightGBM handles its own processing
            print(f"Initialized LightGBM gesture detector")
        else:  # CNN
            self.model = GestureModel(self.config)
            self.video_processor = VideoProcessor(self.config.seq_length)
            print(f"Initialized CNN gesture detector")
            
        self.target_fps = 25.0
    
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
    
    def predict_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """
        Process single video and return predictions.
        Automatically routes to appropriate model implementation.
        """
        if self.model_type == "lightgbm":
            return self._predict_video_lightgbm(video_path, stride)
        else:
            return self._predict_video_cnn(video_path, stride)
    
    def _predict_video_cnn(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray]:
        """Original CNN prediction method."""
        # Extract features and timestamps
        features, timestamps = self.video_processor.process_video(video_path)
    
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
        gesture_class_bias = self.params['gesture_class_bias']
        
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
        results_df['label'] = results_df.apply(
            lambda row: get_prediction_at_threshold(
                row,
                self.params['motion_threshold'],
                self.params['gesture_threshold']
            ),
            axis=1
        )

        # Create segments
        segments = create_segments(
            results_df,
            min_length_s=self.params['min_length_s'],
            label_column='label'
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
                
                # Convert LightGBM output to CNN-compatible format
                if gesture_name == "NOGESTURE":
                    has_motion = 0.0
                    gesture_conf = 0.0
                    move_conf = 0.0
                else:
                    has_motion = confidence  # Use prediction confidence as motion indicator
                    # Distribute confidence based on gesture type
                    if "move" in gesture_name.lower() or "MOVE" in gesture_name:
                        gesture_conf = 0.0
                        move_conf = confidence
                    else:
                        gesture_conf = confidence
                        move_conf = 0.0
                
                predictions.append({
                    'time': timestamp,
                    'has_motion': has_motion,
                    'NoGesture_confidence': 1.0 - has_motion,
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
        results_df = pd.DataFrame(predictions)
        
        if results_df.empty:
            return pd.DataFrame(), {"error": "No predictions generated"}, pd.DataFrame(), np.array([])
        
        # Apply thresholds (reuse existing logic)
        results_df['label'] = results_df.apply(
            lambda row: get_prediction_at_threshold(
                row,
                self.params['motion_threshold'],
                self.params['gesture_threshold']
            ),
            axis=1
        )

        # Create segments
        segments = create_segments(
            results_df,
            min_length_s=self.params['min_length_s'],
            label_column='label'
        )

        # Calculate statistics
        stats = {
            'average_motion': float(results_df['has_motion'].mean()),
            'average_gesture': float(results_df['Gesture_confidence'].mean()),
            'average_move': float(results_df['Move_confidence'].mean()),
            'model_type': self.model_type,
            'lightgbm_features': len(valid_features)
        }
        
        return results_df, stats, segments, np.array(valid_features), valid_timestamps

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
            print(f"\nProcessing {video_name} with {self.model_type.upper()} model...")
            
            try:
                # Process video (automatically routes to correct model)
                print("Extracting features and model inferencing...")
                predictions_df, stats, segments, features, timestamps = self.predict_video(video_path)
                
                if not predictions_df.empty:
                    # Save predictions
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
                        motion_threshold=self.params['motion_threshold'],
                        gesture_threshold=self.params['gesture_threshold'],
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

                    results[video_name] = {
                        "stats": stats,
                        "output_path": output_path
                    }
                    print(f"Done processing {video_name} with {self.model_type.upper()}")
                else:
                    results[video_name] = {"error": "No predictions generated"}
                    
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                results[video_name] = {"error": str(e)}
        
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


class RealtimeGestureDetector:
    """
    Real-time gesture detection class (LightGBM only).
    Provides webcam processing and real-time inference capabilities.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.2,
        config: Optional[Config] = None
    ):
        """Initialize real-time detector with LightGBM model."""
        self.config = config or Config()
        
        # Force LightGBM model
        self.model = LightGBMGestureModel(self.config)
        self.confidence_threshold = confidence_threshold
        
        # Set confidence threshold on model if supported
        if hasattr(self.model, 'set_confidence_threshold'):
            self.model.set_confidence_threshold(confidence_threshold)
        
        print(f"Initialized real-time LightGBM detector")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Advanced features: {'ENABLED' if self.model.includes_fingers else 'DISABLED'}")
    
    def process_webcam(
        self,
        duration: Optional[float] = None,
        camera_index: int = 0,
        show_display: bool = True,
        save_video: bool = False,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process webcam feed in real-time.
        
        Args:
            duration: Maximum duration in seconds (None = unlimited)
            camera_index: Camera device index
            show_display: Whether to show real-time display
            save_video: Whether to save annotated video
            output_csv: Path to save results CSV
            
        Returns:
            DataFrame with frame-by-frame predictions
        """
        print(f"Starting real-time webcam processing...")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Duration: Unlimited (press 'q' to quit)")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_index}")
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera: {width}x{height} at {fps:.1f}fps")
        
        # Setup video writer if requested
        writer = None
        if save_video:
            output_video_path = f"realtime_webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
            print(f"Saving video to: {output_video_path}")
        
        # Reset model state
        self.model.key_joints_buffer.clear()
        if hasattr(self.model, 'left_fingers_buffer'):
            self.model.left_fingers_buffer.clear()
            self.model.right_fingers_buffer.clear()
        
        frame_results = []
        frame_count = 0
        start_time = time.time()
        
        print("Controls: q=quit, spacebar=status, +/-=adjust threshold")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    continue
                
                current_time = time.time() - start_time
                
                # Check duration limit
                if duration and current_time > duration:
                    break
                
                # Extract features and predict
                features = self.model.extract_features_from_frame(frame)
                
                gesture_name = "NoGesture"
                confidence = 0.0
                
                if features is not None:
                    pred_probs = self.model.predict(features.reshape(1, -1))[0]
                    predicted_class = np.argmax(pred_probs)
                    confidence = pred_probs[predicted_class]
                    
                    raw_gesture_name = self.model.label_encoder.inverse_transform([predicted_class])[0]
                    gesture_name = self.model.standardize_gesture_name(raw_gesture_name)
                    
                    # Apply confidence threshold
                    if gesture_name != "NoGesture" and confidence < self.confidence_threshold:
                        gesture_name = "NoGesture"
                        confidence = 0.0
                
                # Store results
                frame_results.append({
                    'frame': frame_count,
                    'timestamp': current_time,
                    'gesture': gesture_name,
                    'confidence': confidence,
                    'threshold': self.confidence_threshold
                })
                
                # Display on frame
                if show_display:
                    display_frame = cv2.flip(frame, 1)  # Mirror effect
                    
                    # Add text overlay
                    color = (0, 255, 0) if gesture_name != "NoGesture" else (128, 128, 128)
                    cv2.putText(display_frame, f"Gesture: {gesture_name}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Threshold: {self.confidence_threshold:.2f}", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Time: {current_time:.1f}s", 
                              (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if writer:
                        writer.write(display_frame)
                    
                    cv2.imshow('Real-time Gesture Detection', display_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):  # Status
                        print(f"Current: {gesture_name} ({confidence:.3f}), Threshold: {self.confidence_threshold:.2f}")
                    elif key == ord('=') or key == ord('+'):  # Increase threshold
                        self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                        if hasattr(self.model, 'set_confidence_threshold'):
                            self.model.set_confidence_threshold(self.confidence_threshold)
                        print(f"Threshold increased to {self.confidence_threshold:.2f}")
                    elif key == ord('-'):  # Decrease threshold
                        self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                        if hasattr(self.model, 'set_confidence_threshold'):
                            self.model.set_confidence_threshold(self.confidence_threshold)
                        print(f"Threshold decreased to {self.confidence_threshold:.2f}")
                
                frame_count += 1
                
                # Periodic status updates
                if frame_count % 1500 == 0:
                    runtime_mins = current_time / 60.0
                    gesture_frames = len([r for r in frame_results if r['gesture'] != 'NoGesture'])
                    gesture_percentage = (gesture_frames / len(frame_results)) * 100 if frame_results else 0
                    print(f"Status: {runtime_mins:.1f}m runtime, {frame_count} frames, {gesture_percentage:.1f}% gestures")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
        
        # Save results
        df = pd.DataFrame(frame_results)
        if output_csv and not df.empty:
            df.to_csv(output_csv, index=False)
            print(f"Results saved to: {output_csv}")
        
        total_time = time.time() - start_time
        print(f"\nReal-time session completed:")
        print(f"Processed {frame_count} frames in {total_time:.1f}s")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        
        if not df.empty:
            gesture_frames = len(df[df['gesture'] != 'NoGesture'])
            print(f"Gestures detected: {gesture_frames} frames ({gesture_frames/len(df)*100:.1f}%)")
        
        return df
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        if hasattr(self.model, 'set_confidence_threshold'):
            self.model.set_confidence_threshold(self.confidence_threshold)
        print(f"Confidence threshold updated to: {self.confidence_threshold}")