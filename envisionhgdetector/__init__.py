"""
EnvisionHGDetector: Hand Gesture Detection Package
"""

from .config import Config
from .detector import GestureDetector
from .model import GestureModel, make_model

__version__ = "1.0.0.1"
__author__ = "Wim Pouw, Bosco Yung, Sharjeel Shaikh, James Trujillo, Antonio Rueda-Toicen, Gerard de Melo, Babajide Owoyele"
__email__ = "wim.pouw@donders.ru.nl"

# Make key classes available at package level
__all__ = [
    'Config',
    'GestureDetector',
    'GestureModel',
    'make_model'
]

# Example usage in docstring
__doc__ = """
EnvisionHGDetector is a package for detecting hand gestures in videos.

Basic usage:
    from envisionhgdetector import GestureDetector
    
    detector = GestureDetector()
    results = detector.process_folder(
        video_folder="path/to/videos",
        output_folder="path/to/output"
    )
    
    from envisionhgdetector import utils
    segments = utils.cut_video_by_segments(outputfolder)

    gesture_segments_folder = os.path.join(outputfolder, "gesture_segments")
    retracked_folder = os.path.join(outputfolder, "retracked")
    analysis_folder = os.path.join(outputfolder, "analysis")
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
"""