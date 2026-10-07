# envisionhgdetector/detector.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from envisionhgdetector.default_config import DefaultConfig
from envisionhgdetector.dashboard.preparation import DashboardMixin
from envisionhgdetector.state import Thresholds, VALID_MODEL_NAMES, ModelNames
from envisionhgdetector.utils import create_elan_file
from envisionhgdetector import GestureModel, BinaryGestureModel, LightGBMGestureModel


# suppress warnings
import logging
logging.getLogger("moviepy").setLevel(logging.WARNING)

class GestureDetector(DashboardMixin):
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

        self.model_type = model_type
        if self.model_type not in VALID_MODEL_NAMES:
            raise ValueError(f"Unknown model type: {model_type}. Use one of {VALID_MODEL_NAMES}.")

        self.config = DefaultConfig(self.model_type, self.thresholds, config_path, weights_path).get_config()

        MODEL_CLASS_MAPPING = {
            ModelNames.CNN: GestureModel,
            ModelNames.CNN_B: BinaryGestureModel,
            ModelNames.LIGHTGBM: LightGBMGestureModel
        }   
        self.model = MODEL_CLASS_MAPPING[self.model_type](self.config)
        print(f"Initialized {self.model_type} gesture detector")
                
    def predict_video(self, video_path: str, stride: int = 1) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, np.ndarray, List[float]]:
        return self.model.predict_video(video_path, stride)  # Call the appropriate model's predict_video method

    def predict_labels_from_landmarks(self, landmarks_per_frame: np.ndarray, fps: float) -> pd.DataFrame:
        return self.model.predict_video_from_landmarks(landmarks_per_frame, fps)
    
    def process_video(self, video_path: str | Path, output_folder: str | Path, elan_only: bool = False):
        print("Elan only flag is set to:", elan_only)
        video_path = Path(video_path)
        output_folder = Path(output_folder)
        output = dict(error=None, stats=None, output_path=None) # elan output path

        if not video_path.exists():
            output["error"] = f"Video not found: {video_path}"
            print(output["error"])
            return output
        
        output_folder.mkdir(parents=True, exist_ok=True)
        video_name = video_path.stem
        video_output_folder = output_folder / video_name
        video_output_folder.mkdir(parents=True, exist_ok=True)

        elan_save_path = video_output_folder / f"{video_name}.eaf"
        segments_save_path = video_output_folder / f"{video_name}_segments.csv"
        predictions_save_path = video_output_folder / f"{video_name}_predictions.csv"
        features_save_path = video_output_folder / f"{video_name}_features.npy"
        print(f"\nProcessing {video_name} with {self.model_type} model...")
        
        try:
            print("Extracting features and model inferencing...")
            predictions_df, stats, segments, features, timestamps = self.predict_video(video_path)
            
            if predictions_df.empty:
                output["error"] = "No predictions generated"
                print(output["error"])
                return output
            
            if not elan_only:
                # Save predictions
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
            
            print("Generating ELAN file...")
            # Create ELAN file
            create_elan_file(
                video_path=video_path,
                segments_df=segments,
                output_path=elan_save_path,
            )

            output['stats'] = stats
            output['output_path'] = elan_save_path
            print(f"Done processing {video_name} with {self.model_type}")

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
