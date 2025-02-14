import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import cv2
import shutil
from .config import Config
from .model import GestureModel
from .preprocessing import VideoProcessor, create_sliding_windows
from .utils import (
    create_segments, get_prediction_at_threshold, create_elan_file, 
    label_video, cut_video_by_segments, retrack_gesture_videos,
    compute_gesture_kinematics_dtw, create_gesture_visualization, create_dashboard,
    setup_dashboard_folders, joint_map, calc_mcneillian_space, calc_vert_height,
    calc_volume_size, calc_holds
)
from .dashboard import run_dashboard_server  # New import for dashboard
from typing import Optional
# Standard library imports
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import statistics

# suppress warnings
import logging

# Suppress MoviePy logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

# We will now also smooth the confidence time series for each gesture class.
def apply_smoothing(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply simple moving average smoothing to a series."""
    return series.rolling(window=window, center=True).mean().fillna(series)
    
class GestureDetector:
    """Main class for gesture detection in videos."""
    
    def __init__(
        self,
        motion_threshold: Optional[float] = None,
        gesture_threshold: Optional[float] = None,
        min_gap_s: Optional[float] = None,
        min_length_s: Optional[float] = None,
        config: Optional[Config] = None
    ):
        """Initialize detector with parameters."""
        self.config = config or Config()
        self.params = {
            'motion_threshold': motion_threshold or self.config.default_motion_threshold,
            'gesture_threshold': gesture_threshold or self.config.default_gesture_threshold,
            'min_gap_s': min_gap_s or self.config.default_min_gap_s,
            'min_length_s': min_length_s or self.config.default_min_length_s
        }
        
        self.model = GestureModel(self.config)
        self.video_processor = VideoProcessor(self.config.seq_length)
    # Add this method after __init__:
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
        
        Args:
            video_path: Path to video file
            stride: Frame stride for sliding windows
            
        Returns:
            DataFrame with predictions and statistics dictionary
        """
        # Extract features
        features, timestamps = self.video_processor.process_video(video_path)
        
        if not features:
            return pd.DataFrame(), {"error": "No features detected"}
        
        # Create windows
        windows = create_sliding_windows(
            features,
            self.config.seq_length,
            stride
        )
        
        if len(windows) == 0:
            return pd.DataFrame(), {"error": "No valid windows created"}
            
        # Get predictions
        predictions = self.model.predict(windows)
        
        # Create results DataFrame
         # get fps
        fps = self._get_video_fps(video_path)
        rows = []
        for i, (pred, time) in enumerate(zip(predictions, timestamps[::stride])):
            has_motion = pred[0]
            gesture_probs = pred[1:]
            
            rows.append({
                'time': time, 
                'has_motion': float(has_motion),
                'NoGesture_confidence': float(1 - has_motion),
                'Gesture_confidence': float(gesture_probs[0]),
                'Move_confidence': float(gesture_probs[1])
            })
        
        results_df = pd.DataFrame(rows)

        # smooth predictions
        #results_df['Gesture_confidence'] = apply_smoothing(results_df['Gesture_confidence'])
        #results_df['Move_confidence'] = apply_smoothing(results_df['Move_confidence'])
        #results_df['has_motion'] = apply_smoothing(results_df['has_motion'])
        
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
            'average_move': float(results_df['Move_confidence'].mean())
        }
        
        return results_df, stats, segments, features
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        video_pattern: str = "*.mp4"
    ) -> Dict[str, Dict]:
        """
        Process all videos in a folder.
        
        Args:
            input_folder: Path to input video folder
            output_folder: Path to output folder
            video_pattern: Pattern to match video files
        """
        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all videos
        videos = glob.glob(os.path.join(input_folder, video_pattern))
        results = {}
        
        for video_path in videos:
            video_name = os.path.basename(video_path)
            print(f"\nProcessing {video_name}...")
            
            try:
                # Process video
                predictions_df, stats, segments, features = self.predict_video(video_path)
                
                if not predictions_df.empty:
                    # Save predictions
                    output_pathpred = os.path.join(
                        output_folder,
                        f"{video_name}_predictions.csv"
                    )
                    predictions_df.to_csv( output_pathpred, index=False)
                    
                    # save segments
                    output_pathseg = os.path.join(
                        output_folder,
                        f"{video_name}_segments.csv"
                    )
                    segments.to_csv(output_pathseg, index=False)

                    # Save features
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
                        output_pathvid
                    )
                    print("Generating elan file...")

                    # Create ELAN file
                    output_path = os.path.join(
                        output_folder,
                        f"{video_name}.eaf"
                    )
                    # get fps
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps = int(fps)
                    cap.release()
                    # Create ELAN file
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
                    # print that were done with this video
                    print(f"Done processing {video_name}, go look in the output folder")
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
        """
        Retrack gesture segments using MediaPipe world landmarks.
        
        Args:
            input_folder: Path to folder containing gesture segments
            output_folder: Path to save retracked results
            
        Returns:
            Dictionary with paths to output files
        """
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
        """
        Compute DTW distances, kinematic features, and create visualization.
        
        Args:
            landmarks_folder: Folder containing world landmarks data
            output_folder: Path to save analysis results
            fps: Frames per second of the video
            
        Returns:
            Dictionary with paths to output files
        """
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
    
    # Conditionally import JupyterDash if running in a Jupyter notebook environment
    def prepare_gesture_dashboard(self, data_folder: str, assets_folder: Optional[str] = None) -> None:
        """
        Prepare folders and copy the app.py for user to run in a standalone Python environment.
        Also creates a CSS file in the assets folder.
        """
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
            
            # Optionally, inform the user to run the app.py
            print("Run 'python app.py' to start the dashboard in a Python environment.")
            
        except Exception as e:
            print(f"Error preparing dashboard: {str(e)}")
            raise


