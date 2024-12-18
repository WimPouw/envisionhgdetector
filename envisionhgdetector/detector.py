import os
import glob
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from .config import Config
from .model import GestureModel
from .preprocessing import VideoProcessor, create_sliding_windows

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
        rows = []
        for i, pred in enumerate(predictions):
            has_motion = pred[0]
            gesture_probs = pred[1:]
            
            rows.append({
                'time': i * stride / 30,  # Assuming 30 fps
                'has_motion': float(has_motion),
                'NoGesture_confidence': float(1 - has_motion),
                'Gesture_confidence': float(gesture_probs[0]),
                'Move_confidence': float(gesture_probs[1])
            })
        
        results_df = pd.DataFrame(rows)
        
        # Calculate statistics
        stats = {
            'average_motion': float(results_df['has_motion'].mean()),
            'average_gesture': float(results_df['Gesture_confidence'].mean()),
            'average_move': float(results_df['Move_confidence'].mean())
        }
        
        return results_df, stats
    
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
                predictions_df, stats = self.predict_video(video_path)
                
                if not predictions_df.empty:
                    # Save predictions
                    output_path = os.path.join(
                        output_folder,
                        f"{video_name}_predictions.csv"
                    )
                    predictions_df.to_csv(output_path, index=False)
                    
                    results[video_name] = {
                        "stats": stats,
                        "output_path": output_path
                    }
                else:
                    results[video_name] = {"error": "No predictions generated"}
                    
            except Exception as e:
                print(f"Error processing {video_name}: {str(e)}")
                results[video_name] = {"error": str(e)}
        
        return results