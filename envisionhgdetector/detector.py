import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .config import Config
from .model import GestureModel
from .preprocessing import VideoProcessor, create_sliding_windows
from .utils import create_segments, get_prediction_at_threshold, create_elan_file, label_video, cut_video_by_segments
import cv2 as cv2

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
    
    def predict_video(
        self,
        video_path: str,
        stride: int = 1
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Process single video and return predictions.
        
        Args:
            video_path: Path to video file
            stride: Frame stride for sliding windows
            
        Returns:
            DataFrame with predictions and statistics dictionary
        """
        # Extract features
        features = self.video_processor.process_video(video_path)
        
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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps)
        cap.release()
        rows = []
        for i, pred in enumerate(predictions):
            has_motion = pred[0]
            gesture_probs = pred[1:]
            
            rows.append({
                'time': i * stride / fps,  # Assuming 30 fps
                'has_motion': float(has_motion),
                'NoGesture_confidence': float(1 - has_motion),
                'Gesture_confidence': float(gesture_probs[0]),
                'Move_confidence': float(gesture_probs[1])
            })
        
        results_df = pd.DataFrame(rows)

        # smooth predictions
        results_df['Gesture_confidence'] = apply_smoothing(results_df['Gesture_confidence'])
        results_df['Move_confidence'] = apply_smoothing(results_df['Move_confidence'])
        results_df['has_motion'] = apply_smoothing(results_df['has_motion'])
        
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
        