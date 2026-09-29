# envisionhgdetector/detector.py

import os
import glob
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from envisionhgdetector import GestureModel  # Renamed CNN model
from envisionhgdetector import BinaryGestureModel  # New binary CNN model
from envisionhgdetector import LightGBMGestureModel  # New LightGBM model
from envisionhgdetector.default_config import DefaultConfig
from envisionhgdetector.state import ModelNames, Thresholds, Labels, VALID_MODEL_NAMES, VALID_MODEL_NAMES_LITERAL, MoveMode
from envisionhgdetector.utils import (
    create_elan_file, 
    create_segments_from_labels,
    label_video, retrack_gesture_videos,
    compute_gesture_kinematics_dtw, create_gesture_visualization,
    setup_dashboard_folders,
    get_video_fps
)

# suppress warnings
import logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

def apply_smoothing(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply simple moving average smoothing to a series."""
    return series.rolling(window=window, center=True).mean().fillna(series)

class GestureDetector:
    """Main class for gesture detection in videos - supports CNN, LightGBM, and Combined models."""
    def __init__(
        self,
        model_type: str,
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
        
        if self.model_type == ModelNames.LIGHTGBM:
            self.config = DefaultConfig("lightgbm", self.thresholds, config_path, weights_path).get_config()
            self.model = LightGBMGestureModel(self.config)
            print(f"Initialized LightGBM gesture detector")

        elif self.model_type == ModelNames.CNN_B:
            self.config = DefaultConfig("cnn_b", self.thresholds, config_path, weights_path).get_config()
            self.model = BinaryGestureModel(self.config)
            print(f"Initialized CNN-B gesture detector")

        else:  # CNN
            self.config = DefaultConfig("cnn", self.thresholds, config_path, weights_path).get_config()
            self.model = GestureModel(self.config)
            print(f"Initialized CNN gesture detector")
                

    def predict_video(self, video_path: str, stride: int = 1) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray, List[float]]:
        return self.model.predict_video(video_path, stride)  # Call the appropriate model's predict_video method

    def predict_labels_from_landmarks(self, landmarks_per_frame: np.ndarray, fps: float) -> pd.DataFrame:
        results = self.model.predict_video_from_landmarks(landmarks_per_frame, fps)
        return results

    # def _create_segments_from_predictions(
    #     self, 
    #     raw_df: pd.DataFrame, 
    #     class_column: str,
    #     threshold: float
    # ) -> pd.DataFrame:
    #     """Create segments from predictions using the shared timestamp helper."""
    #     columns = ['start_time', 'end_time', 'prediction', 'prediction_id', 'duration']
    #     if raw_df.empty or class_column not in raw_df.columns:
    #         return pd.DataFrame(columns=columns)

    #     segments = create_segments_from_labels(
    #         raw_df['timestamp'].to_numpy(),
    #         raw_df[class_column].tolist(),
    #         min_gap_s=self.config.thresholds.min_gap_s,
    #         min_length_s=self.config.thresholds.min_length_s,
    #         move_mode=MoveMode.AS_GESTURE,
    #     )
    #     return segments
    
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
                    print(f"Saved predictions to {output_pathpred}")
                    
                    # Save segments
                    output_pathseg = os.path.join(
                        output_folder,
                        f"{video_name}_segments.csv"
                    )
                    segments.to_csv(output_pathseg, index=False)
                    print(f"Saved segments to {output_pathseg}")

                    # Save features (if available)
                    if len(features) > 0:
                        output_pathfeat = os.path.join(
                            output_folder,
                            f"{video_name}_features.npy"
                        )
                        feature_array = np.array(features)
                        np.save(output_pathfeat, feature_array)
                        print(f"Saved features to {output_pathfeat}")

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
                fps = get_video_fps(video_path)
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

