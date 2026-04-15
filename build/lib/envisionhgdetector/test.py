from detector import GestureDetector
import utils

import os

# Absolute paths (recommended)
videofoldertoday = os.path.abspath('test')
outputfolder = os.path.abspath('output')

# Create detector with combined model
detector = GestureDetector(
    model_type="combined",
    cnn_motion_threshold=0.5,    # Motion gate sensitivity
    cnn_gesture_threshold=0.5,   # CNN gesture confidence
    lgbm_threshold=0.5,          # LightGBM gesture probability
    min_gap_s=0.1,               # Merge gaps smaller than this
    min_length_s=0.1             # Minimum gesture duration
)

detector.process_folder(
    input_folder=videofoldertoday,
    output_folder=outputfolder,
)

segments = utils.cut_video_by_segments(outputfolder)

# Check the gesture segments folder
gesture_segments_folder = os.path.join(outputfolder, "gesture_segments")
if os.path.exists(gesture_segments_folder):
    segment_files = [f for f in os.listdir(gesture_segments_folder) if f.endswith('.mp4')]
    print(f"Found {len(segment_files)} gesture segment files")

gesture_segments_folder = os.path.join(outputfolder, "gesture_segments")
retracked_folder = os.path.join(outputfolder, "retracked")
analysis_folder = os.path.join(outputfolder, "analysis")

print("Step 4: Retracking gestures with world landmarks...")
tracking_results = detector.retrack_gestures(
    input_folder=gesture_segments_folder,
    output_folder=retracked_folder
)
print(f"Tracking results: {tracking_results}")

if "error" not in tracking_results:
    print("Step 5: Computing DTW and kinematics...")
    analysis_results = detector.analyze_dtw_kinematics(
        landmarks_folder=tracking_results["landmarks_folder"],
        output_folder=analysis_folder
    )
    print(f"Analysis results: {analysis_results}")

if "error" not in analysis_results:
    print("Step 6: Preparing dashboard...")
    detector.prepare_gesture_dashboard(
        data_folder=analysis_folder
    )