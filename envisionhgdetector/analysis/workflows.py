"""Analysis workflows exposed by gesture detectors."""

import os
from typing import Dict

from .retracking import retrack_gesture_videos
from .dtw import compute_gesture_kinematics_dtw
from .visualization import create_gesture_visualization


class AnalysisMixin:
    """Analysis operations shared by detector implementations."""

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
