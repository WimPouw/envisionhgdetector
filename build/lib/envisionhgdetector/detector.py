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
    Provides webcam processing and real-time inference capabilities with post-processing.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.2,
        min_gap_s: float = 0.3,          # Fixed - no real-time adjustment
        min_length_s: float = 0.5,       # Fixed - no real-time adjustment
        config: Optional[Config] = None
    ):
        """Initialize real-time detector with LightGBM model and refinement parameters."""
        self.config = config or Config()
        
        # Force LightGBM model
        self.model = LightGBMGestureModel(self.config)
        
        # Validate and store parameters with proper defaults
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.min_gap_s = max(0.0, float(min_gap_s))
        self.min_length_s = max(0.0, float(min_length_s))
        
        # Set confidence threshold on model if supported
        if hasattr(self.model, 'set_confidence_threshold'):
            self.model.set_confidence_threshold(self.confidence_threshold)
        
        print(f"Initialized real-time LightGBM detector")
        print(f"Confidence threshold: {self.confidence_threshold:.2f} (fixed)")
        print(f"Min gap between gestures: {self.min_gap_s:.2f}s (fixed)")
        print(f"Min gesture length: {self.min_length_s:.2f}s (fixed)")
        print(f"Advanced features: {'ENABLED' if self.model.includes_fingers else 'DISABLED'}")
        
        # Debug: verify parameters are set
        assert hasattr(self, 'confidence_threshold'), "confidence_threshold not set"
        assert hasattr(self, 'min_gap_s'), "min_gap_s not set"
        assert hasattr(self, 'min_length_s'), "min_length_s not set"
        print(f"Parameter validation passed.")
        
    def process_webcam(
        self,
        duration: Optional[float] = None,
        camera_index: int = 0,
        show_display: bool = True,
        save_video: bool = True,
        apply_post_processing: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process webcam feed in real-time with post-processing.
        
        Args:
            duration: Maximum duration in seconds (None = unlimited)
            camera_index: Camera device index
            show_display: Whether to show real-time display
            save_video: Whether to save annotated video
            apply_post_processing: Whether to apply segment refinement
            
        Returns:
            Tuple of (raw_results_df, segments_df)
        """
        print(f"Starting real-time webcam processing...")
        if duration:
            print(f"Duration: {duration} seconds")
        else:
            print("Duration: Unlimited (press 'q' to quit)")
        
        # Create output folder structure
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_base = "output_realtime"
        session_folder = os.path.join(output_base, f"session_{timestamp}")
        os.makedirs(session_folder, exist_ok=True)
        
        print(f"Output folder: {session_folder}")
        
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
        video_path = None
        output_fps = 20.0  # Define output FPS as a variable
        if save_video:
            video_path = os.path.join(session_folder, f"webcam_session.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, output_fps, (width, height))
            print(f"Saving video to: {video_path} at {output_fps} FPS")
        
        # Reset model state
        self.model.key_joints_buffer.clear()
        if hasattr(self.model, 'left_fingers_buffer'):
            self.model.left_fingers_buffer.clear()
            self.model.right_fingers_buffer.clear()
        
        frame_results = []
        frame_count = 0
        start_time = time.time()
        
        print("\nControls:")
        print("  - Q: Quit session")
        print("  - SPACE: Show current status")
        print()
        
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
                    
                    # Apply confidence threshold (fixed)
                    if gesture_name != "NoGesture" and confidence < self.confidence_threshold:
                        gesture_name = "NoGesture"
                        confidence = 0.0
                
                # Calculate frame-based timestamp that matches video output
                # This ensures ELAN timestamps align with video frames
                # Use frame_count/fps for video sync, wall clock for user display
                video_timestamp = frame_count / output_fps if save_video else current_time
                
                # Store results with both timestamps
                frame_results.append({
                    'frame': frame_count,
                    'timestamp': video_timestamp,  # Video-aligned timestamp for ELAN
                    'wall_clock_time': current_time,  # Real time for user feedback
                    'gesture': gesture_name,
                    'confidence': confidence,
                    'threshold': self.confidence_threshold,
                    'raw_gesture': gesture_name
                })
                
                # Display on frame
                if show_display:
                    display_frame = cv2.flip(frame, 1)  # Mirror effect
                    
                    # Add text overlay (use wall clock time for display)
                    color = (0, 255, 0) if gesture_name != "NoGesture" else (128, 128, 128)
                    cv2.putText(display_frame, f"Gesture: {gesture_name}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.putText(display_frame, f"Confidence: {confidence:.2f}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display_frame, f"Time: {current_time:.1f}s", 
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Frame: {frame_count}", 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Save frame if requested
                    if writer:
                        writer.write(display_frame)
                    
                    cv2.imshow('Real-time Gesture Detection', display_frame)
                    
                    # Handle keyboard input (simplified)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q') or key == ord('Q'):
                        print("Quit requested")
                        break
                    elif key == ord(' '):  # Status
                        print(f"Current: {gesture_name} ({confidence:.3f})")
                        print(f"Parameters: threshold={self.confidence_threshold:.2f}, gap={self.min_gap_s:.1f}s, minlen={self.min_length_s:.1f}s")
                
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
        
        # Convert to DataFrame
        raw_df = pd.DataFrame(frame_results)
        
        if raw_df.empty:
            print("No data recorded")
            return pd.DataFrame(), pd.DataFrame()
        
        # Save raw results
        raw_csv_path = os.path.join(session_folder, "raw_frame_results.csv")
        raw_df.to_csv(raw_csv_path, index=False)
        print(f"Raw results saved to: {raw_csv_path}")
        
        # Debug timing information
        if save_video and not raw_df.empty:
            print(f"Timing alignment info:")
            print(f"   Video duration: {raw_df['timestamp'].max():.1f}s (based on {output_fps} FPS)")
            print(f"   Wall clock duration: {raw_df['wall_clock_time'].max():.1f}s")
            print(f"   Frame count: {len(raw_df)} frames")
            print(f"   Expected video duration: {len(raw_df) / output_fps:.1f}s")
        
        # Apply post-processing if requested
        segments_df = pd.DataFrame()
        if apply_post_processing:
            try:
                print("Applying post-processing segmentation...")
                
                # Create processed dataframe 
                processed_df = raw_df.copy()
                processed_df['time'] = processed_df['timestamp']  # Required for segmentation
                processed_df['original_gesture'] = processed_df['gesture']  # Keep original gesture names
                
                # Count gesture vs non-gesture frames
                gesture_frames = processed_df[processed_df['gesture'].apply(
                    lambda x: x not in ['NoGesture', 'NOGESTURE']
                )].shape[0]
                total_frames = len(processed_df)
                
                print(f"Frame analysis:")
                print(f"  Total frames: {total_frames}")
                print(f"  Gesture frames: {gesture_frames} ({gesture_frames/total_frames*100:.1f}%)")
                print(f"  Post-processing parameters: min_gap={self.min_gap_s:.2f}s, min_length={self.min_length_s:.2f}s")
                
                # Apply segmentation
                segments_df = self._create_gesture_segments(processed_df)
                
                if not segments_df.empty:
                    # Save processed segments
                    segments_csv_path = os.path.join(session_folder, "gesture_segments.csv")
                    segments_df.to_csv(segments_csv_path, index=False)
                    print(f"Segments saved to: {segments_csv_path}")
                    
                    # Create ELAN file - only if video was saved
                    if save_video and video_path and os.path.exists(video_path):
                        try:
                            print("Creating ELAN file...")
                            elan_path = os.path.join(session_folder, "gesture_segments.eaf")
                            
                            try:
                                from .utils import create_elan_file
                            except ImportError:
                                from utils import create_elan_file
                                
                            create_elan_file(
                                video_path=video_path,
                                segments_df=segments_df,
                                output_path=elan_path,
                                fps=output_fps,
                                include_ground_truth=False
                            )
                            print(f"ELAN file saved to: {elan_path}")
                        except Exception as e:
                            print(f"Error creating ELAN file: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("Skipping ELAN creation (video not saved or not found)")
                    
                    # Print summary
                    total_segments = len(segments_df)
                    total_gesture_time = segments_df['duration'].sum()
                    avg_segment_length = segments_df['duration'].mean()
                    
                    print(f"\nPost-processing Summary:")
                    print(f"   Gesture segments created: {total_segments}")
                    print(f"   Total gesture time: {total_gesture_time:.1f}s")
                    print(f"   Average segment duration: {avg_segment_length:.1f}s")
                    if 'wall_clock_time' in raw_df.columns:
                        total_time = raw_df['wall_clock_time'].max()
                        print(f"   Gestures per minute: {total_segments / (total_time / 60):.1f}")
                else:
                    print("\nNo gesture segments found after post-processing")
                    print("Suggestions:")
                    print(f"- Try reducing min_length (current: {self.min_length_s:.2f}s)")
                    print(f"- Try increasing min_gap (current: {self.min_gap_s:.2f}s)")
                    print("- Check if gestures are being detected consistently in the video")
                    
            except Exception as e:
                print(f"Error during post-processing: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save session summary
        self._save_session_summary(session_folder, raw_df, segments_df)
        
        total_time = time.time() - start_time
        print(f"\nReal-time session completed:")
        print(f"   Processed {frame_count} frames in {total_time:.1f}s")
        print(f"   Average FPS: {frame_count/total_time:.1f}")
        print(f"   Session folder: {session_folder}")
        
        if not raw_df.empty:
            gesture_frames = len(raw_df[raw_df['gesture'] != 'NoGesture'])
            print(f"   Raw gestures detected: {gesture_frames} frames ({gesture_frames/len(raw_df)*100:.1f}%)")
        
        return raw_df, segments_df

    def _create_gesture_segments(self, processed_df):
        """
        Create gesture segments from captured frame data using standard segmentation logic.
        This runs AFTER capture is complete, not during real-time processing.
        """
        import pandas as pd
        import numpy as np
        
        # Create binary gesture indicator (anything not NoGesture is a gesture)
        is_gesture = processed_df['original_gesture'].apply(
            lambda x: x not in ['NoGesture', 'NOGESTURE']
        ).astype(int)
        
        # Find state changes
        changes = np.diff(is_gesture, prepend=0)
        start_indices = np.where(changes == 1)[0]  # Start of gesture periods
        end_indices = np.where(changes == -1)[0]   # End of gesture periods
        
        # Handle case where recording ends during a gesture
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(processed_df) - 1)
        
        print(f"Found {len(start_indices)} potential gesture periods before filtering")
        
        # Create segments with gap merging and minimum length filtering
        segments = []
        segment_id = 1
        
        for i, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
            start_time = processed_df.iloc[start_idx]['time']
            end_time = processed_df.iloc[end_idx]['time']
            duration = end_time - start_time
            
            # Apply minimum length filter
            if duration >= self.min_length_s:
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'label': 'Gesture',
                    'labelid': segment_id,
                    'duration': duration
                })
                segment_id += 1
        
        # Apply gap merging if we have multiple segments
        if len(segments) > 1:
            merged_segments = []
            current_segment = segments[0]
            
            for next_segment in segments[1:]:
                gap = next_segment['start_time'] - current_segment['end_time']
                
                # If gap is smaller than min_gap_s, merge segments
                if gap <= self.min_gap_s:
                    current_segment['end_time'] = next_segment['end_time']
                    current_segment['duration'] = current_segment['end_time'] - current_segment['start_time']
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            
            # Add the last segment
            merged_segments.append(current_segment)
            segments = merged_segments
            
            print(f"After gap merging (gap<={self.min_gap_s:.2f}s): {len(segments)} segments")
        
        # Convert to DataFrame
        if segments:
            segments_df = pd.DataFrame(segments)
            print(f"Final gesture segments: {len(segments_df)}")
            
            # Print details
            for idx, seg in segments_df.iterrows():
                print(f"  Segment {idx+1}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s ({seg['duration']:.2f}s)")
                
            return segments_df
        else:
            print("No gesture segments found after applying filters")
            return pd.DataFrame(columns=['start_time', 'end_time', 'labelid', 'label', 'duration'])
    
    def _save_session_summary(self, session_folder: str, raw_df: pd.DataFrame, segments_df: pd.DataFrame):
        """Save a summary of the session parameters and results as CSV."""
        import pandas as pd
        
        # Create flattened summary data for CSV
        summary_data = {
            # Session info
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_frames': len(raw_df),
            'duration_seconds': raw_df['timestamp'].max() if not raw_df.empty else 0,
            'wall_clock_duration_seconds': raw_df['wall_clock_time'].max() if not raw_df.empty and 'wall_clock_time' in raw_df.columns else 0,
            'average_fps': len(raw_df) / raw_df['timestamp'].max() if not raw_df.empty and raw_df['timestamp'].max() > 0 else 0,
            
            # Parameters
            'confidence_threshold': self.confidence_threshold,
            'min_gap_s': self.min_gap_s,
            'min_length_s': self.min_length_s,
            'model_type': 'LightGBM',
            'advanced_features': self.model.includes_fingers,
            
            # Results
            'raw_gesture_frames': len(raw_df[raw_df['gesture'] != 'NoGesture']) if not raw_df.empty else 0,
            'raw_gesture_percentage': (len(raw_df[raw_df['gesture'] != 'NoGesture']) / len(raw_df) * 100) if not raw_df.empty else 0,
            'processed_segments': len(segments_df) if not segments_df.empty else 0,
            'total_gesture_time': segments_df['duration'].sum() if not segments_df.empty else 0,
            'average_segment_duration': segments_df['duration'].mean() if not segments_df.empty else 0,
            'gestures_per_minute': (len(segments_df) / (raw_df['wall_clock_time'].max() / 60)) if not raw_df.empty and 'wall_clock_time' in raw_df.columns and raw_df['wall_clock_time'].max() > 0 else 0
        }
        
        # Convert to DataFrame with single row
        summary_df = pd.DataFrame([summary_data])
        
        # Save as CSV
        summary_path = os.path.join(session_folder, "session_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Session summary saved to: {summary_path}")
        
        # Also save a more detailed version with individual segment information if segments exist
        if not segments_df.empty:
            detailed_summary = []
            for idx, segment in segments_df.iterrows():
                detailed_row = summary_data.copy()  # Include all session info
                detailed_row.update({
                    'segment_id': segment['labelid'],
                    'segment_label': segment['label'],
                    'segment_start_time': segment['start_time'],
                    'segment_end_time': segment['end_time'],
                    'segment_duration': segment['duration']
                })
                detailed_summary.append(detailed_row)
            
            detailed_df = pd.DataFrame(detailed_summary)
            detailed_path = os.path.join(session_folder, "session_summary_detailed.csv")
            detailed_df.to_csv(detailed_path, index=False)
            
            print(f"Detailed session summary saved to: {detailed_path}")
    
    def load_and_analyze_session(self, session_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and analyze a previous session."""
        raw_csv = os.path.join(session_folder, "raw_frame_results.csv")
        segments_csv = os.path.join(session_folder, "gesture_segments.csv")
        summary_json = os.path.join(session_folder, "session_summary.csv")
        
        raw_df = pd.DataFrame()
        segments_df = pd.DataFrame()
        
        if os.path.exists(raw_csv):
            raw_df = pd.read_csv(raw_csv)
            print(f"Loaded raw results: {len(raw_df)} frames")
        
        if os.path.exists(segments_csv):
            segments_df = pd.read_csv(segments_csv)
            print(f"Loaded segments: {len(segments_df)} segments")
        
        if os.path.exists(summary_json):
            import json
            with open(summary_json, 'r') as f:
                summary = json.load(f)
            print(f"Session summary:")
            print(f"   Duration: {summary['session_info']['duration_seconds']:.1f}s")
            print(f"   Parameters: threshold={summary['parameters']['confidence_threshold']:.2f}")
            print(f"   Results: {summary['results']['processed_segments']} segments")
        
        return raw_df, segments_df
    
    def set_refinement_parameters(self, min_gap_s: float = None, min_length_s: float = None):
        """Update post-processing refinement parameters."""
        if min_gap_s is not None:
            self.min_gap_s = max(0.1, min(2.0, min_gap_s))
            print(f"Min gap updated to: {self.min_gap_s}s")
        
        if min_length_s is not None:
            self.min_length_s = max(0.1, min(3.0, min_length_s))
            print(f"Min length updated to: {self.min_length_s}s")
    
    def load_and_analyze_session(self, session_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and analyze a previous session."""
        raw_csv = os.path.join(session_folder, "raw_frame_results.csv")
        segments_csv = os.path.join(session_folder, "gesture_segments.csv")
        summary_json = os.path.join(session_folder, "session_summary.csv")
        
        raw_df = pd.DataFrame()
        segments_df = pd.DataFrame()
        
        if os.path.exists(raw_csv):
            raw_df = pd.read_csv(raw_csv)
            print(f"Loaded raw results: {len(raw_df)} frames")
        
        if os.path.exists(segments_csv):
            segments_df = pd.read_csv(segments_csv)
            print(f"Loaded segments: {len(segments_df)} segments")
        
        if os.path.exists(summary_json):
            import json
            with open(summary_json, 'r') as f:
                summary = json.load(f)
            print(f"Session summary:")
            print(f"   Duration: {summary['session_info']['duration_seconds']:.1f}s")
            print(f"   Parameters: threshold={summary['parameters']['confidence_threshold']:.2f}")
            print(f"   Results: {summary['results']['processed_segments']} segments")
        
        return raw_df, segments_df