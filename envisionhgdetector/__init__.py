"""
EnvisionHGDetector: Hand Gesture Detection Package
Supports both CNN and LightGBM models for gesture detection.
"""

from .config import Config
from .detector import GestureDetector, RealtimeGestureDetector

__version__ = "2.0.0.0"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele"
__email__ = "wim.pouw@donders.ru.nl"

# Make key classes available at package level
__all__ = ['Config', 'GestureDetector', 'RealtimeGestureDetector']

# Example usage in docstring
__doc__ = """
EnvisionHGDetector is a package for detecting hand gestures in videos using CNN or LightGBM models.

Basic usage with CNN model:
    from envisionhgdetector import GestureDetector
    
    detector = GestureDetector(model_type="cnn")
    results = detector.process_folder(
        input_folder="path/to/videos",
        output_folder="path/to/output"
    )

Basic usage with LightGBM model:
    from envisionhgdetector import GestureDetector
    
    detector = GestureDetector(model_type="lightgbm")
    results = detector.process_folder(
        input_folder="path/to/videos", 
        output_folder="path/to/output"
    )

Real-time detection (LightGBM only):
    from envisionhgdetector import RealtimeGestureDetector
    
    realtime_detector = RealtimeGestureDetector(confidence_threshold=0.3)
    results_df = realtime_detector.process_webcam(duration=60)

Post-processing and analysis:
    from envisionhgdetector import utils
    segments = utils.cut_video_by_segments(output_folder)

    gesture_segments_folder = os.path.join(output_folder, "gesture_segments")
    retracked_folder = os.path.join(output_folder, "retracked")
    analysis_folder = os.path.join(output_folder, "analysis")
    
    tracking_results = detector.retrack_gestures(
        input_folder=gesture_segments_folder,
        output_folder=retracked_folder
    )

    analysis_results = detector.analyze_dtw_kinematics(
        landmarks_folder=tracking_results["landmarks_folder"],
        output_folder=analysis_folder
    )

    detector.prepare_gesture_dashboard(
        data_folder=analysis_folder
    )

Model availability check:
    from envisionhgdetector import Config
    
    config = Config()
    print(f"CNN available: {config.validate_model_availability('cnn')}")
    print(f"LightGBM available: {config.validate_model_availability('lightgbm')}")
    print(config)  # Shows available models with checkmarks
"""

def get_available_models():
    """Get a list of available model types."""
    config = Config()
    available = []
    
    if config.validate_model_availability("cnn"):
        available.append("cnn")
    if config.validate_model_availability("lightgbm"):
        available.append("lightgbm")
    
    return available

def print_model_status():
    """Print the status of available models."""
    config = Config()
    print("EnvisionHGDetector Model Status:")
    print("=" * 35)
    print(f"CNN Model:     {'✓ Available' if config.validate_model_availability('cnn') else '✗ Not Found'}")
    print(f"LightGBM Model: {'✓ Available' if config.validate_model_availability('lightgbm') else '✗ Not Found'}")
    print(f"Config: {config}")
    
    if config.weights_path:
        print(f"\nCNN Model Path: {config.weights_path}")
    if config.lightgbm_model_path:
        print(f"LightGBM Model Path: {config.lightgbm_model_path}")