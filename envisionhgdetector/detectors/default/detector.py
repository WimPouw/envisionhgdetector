# envisionhgdetector/detector.py

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from envisionhgdetector import GestureModel  # Renamed CNN model
from envisionhgdetector import BinaryGestureModel  # New binary CNN model
from envisionhgdetector import LightGBMGestureModel  # New LightGBM model
from envisionhgdetector.default_config import DefaultConfig
from envisionhgdetector.analysis.workflows import AnalysisMixin
from envisionhgdetector.dashboard.preparation import DashboardMixin
from envisionhgdetector.state import ModelNames, Thresholds, Labels, VALID_MODEL_NAMES, VALID_MODEL_NAMES_LITERAL, MoveMode
from envisionhgdetector.utils import (
    create_elan_file, 
    create_segments_from_labels,
    label_video,
    get_video_fps
)

# suppress warnings
import logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

def apply_smoothing(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply simple moving average smoothing to a series."""
    return series.rolling(window=window, center=True).mean().fillna(series)

class GestureDetector(AnalysisMixin, DashboardMixin):
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
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        output = dict(error=None, stats=None, output_path=None) # elan output path

        if not video_path.exists():
            output["error"] = f"Video not found: {video_path}"
            print(output["error"])
            return output
        
        output_folder.mkdir(parents=True, exist_ok=True)
        video_name, video_extension = video_path.stem, video_path.suffix
        video_output_folder = output_folder / video_name
        video_output_folder.mkdir(parents=True, exist_ok=True)

        elan_save_path = video_output_folder / f"{video_name}.eaf"
        segments_save_path = video_output_folder / f"{video_name}_segments.csv"
        predictions_save_path = video_output_folder / f"{video_name}_predictions.csv"
        features_save_path = video_output_folder / f"{video_name}_features.npy"
        labeled_video_path = video_output_folder / f"{video_name}_labeled{video_extension}"
        print(f"\nProcessing {video_name} with {self.model_type} model...")
        
        try:
            print("Extracting features and model inferencing...")
            predictions_df, stats, segments, features, timestamps = self.predict_video(video_path)
            
            if not predictions_df.empty:
                # Save predictions
                if not elan_only:
                    predictions_df.to_csv(predictions_save_path, index=False)
                    print(f"Saved predictions to {predictions_save_path}")
                    
                    # Save segments
                    segments.to_csv(segments_save_path, index=False)
                    print(f"Saved segments to {segments_save_path}")

                    # Save features (if available)
                    if len(features) > 0:
                        feature_array = np.array(features)
                        np.save(features_save_path, feature_array)
                        print(f"Saved features to {features_save_path}")

                # Labeled video generation
                    print("Generating labeled video...")
                    label_video(
                        str(video_path), 
                        segments, 
                        str(labeled_video_path),
                        predictions_df,
                        valid_timestamps=timestamps,
                        motion_threshold=self.model.config.thresholds.motion_threshold,
                        gesture_threshold=self.model.config.thresholds.gesture_threshold,
                        target_fps=getattr(self.model.config, "target_fps", None)
                    )
                
                print("Generating ELAN file...")
                # Create ELAN file
                fps = get_video_fps(video_path)
                create_elan_file(
                    video_path,
                    segments,
                    elan_save_path,
                    fps=fps,
                    include_ground_truth=False
                )

                output['stats'] = stats
                output['output_path'] = elan_save_path
                print(f"Done processing {video_name} with {self.model_type}")
            else:
                output["error"] = "No predictions generated"

        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            output["error"] =  str(e)
        
        return output
    
    def process_folder(self, input_folder: str, output_folder: str, video_pattern: str = ".mp4") -> Dict[str, Dict]:
        """Process all videos in a folder (works with both CNN and LightGBM)."""
        # Create output directories
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        results = {}
        
        # Get all videos
        videos = [
            path
            for path in input_folder.rglob(video_pattern)
            if path.is_file()
        ]
        
        for video_path in videos:
            results[video_path.stem] = self.process_video(video_path, output_folder)
            
        return results
