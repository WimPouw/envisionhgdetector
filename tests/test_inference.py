import numpy as np
from pathlib import Path

from envisionhgdetector import GestureDetector, CombinedGestureDetector, RealtimeGestureDetector
from envisionhgdetector import label_video

def main():
    video_path = Path("path/to/input/video.mp4")  # Replace with the desired video file path
    npz_path = Path("path/to/input/data.npz")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file {npz_path} does not exist.")

    output_folder = Path("path/to/output/folder")  # Replace with the desired output folder path
    output_folder.mkdir(parents=True, exist_ok=True)

    # Uncomment if you only want to test one functionality.
    # video_path = None # skip video testing for now
    # npz_path = None # skip npz testing for now

    # Comment any line to skip the detector
    test_gesture_detector(output_folder, model_name="cnn_b", video_path=video_path, npz_path=npz_path)
    test_gesture_detector(output_folder, model_name="lightgbm", video_path=video_path, npz_path=npz_path)
    test_combined_gesture_detector(output_folder, video_path=video_path, npz_path=npz_path)
    test_realtime_gesture_detector()

def test_video(video_path: str, output_folder: str, detector, name:str):
    print(f"testing video: {video_path} with detector: {name} saving results to: {output_folder}")
    result = detector.process_video(
        video_path,
        output_folder,
    ) 
    output_folder = Path(output_folder) / video_path.stem
    label_video(video_path=video_path, video_output_folder=str(output_folder))

def test_npz(npz_path: str, output_folder: str, detector, name:str):
    print(f"testing npz: {npz_path} with detector: {name} saving results to: {output_folder}")
    file_contents = np.load(npz_path, allow_pickle=False)
    fps = file_contents['fps'].item()  # Extract fps from the npz file
    landmarks = file_contents['world_body_landmarks']

    UPPER_BODY_INDICES = list(range(23)) # we are only interested in the upper body landmarks for gesture detection
    all_features = []

    for frame_landmarks in landmarks: # per frame
        features = []
        for idx in UPPER_BODY_INDICES: # per landmark
            if idx < len(frame_landmarks):
                lm = frame_landmarks[idx, :]
                features.extend(lm)

        all_features.append(features)

    stripped_features = np.array(all_features, dtype=np.float32)
    (output_folder / npz_path.stem).mkdir(parents=True, exist_ok=True)

    results_df = detector.predict_labels_from_landmarks(stripped_features, fps=fps)
    results_df.to_csv(output_folder / npz_path.stem / f"{npz_path.stem}_predictions.csv", index=False) # save results to csv

def test_combined_gesture_detector(output_folder: str, video_path: str = None, npz_path: str = None):
    output_folder = output_folder / "combined_results"
    output_folder.mkdir(parents=True, exist_ok=True)

    detector = CombinedGestureDetector()
    if video_path:
        test_video(video_path, output_folder, detector, name="Combined")
    if npz_path:
        test_npz(npz_path, output_folder, detector, name="Combined")

def test_gesture_detector(output_folder: str, model_name: str, video_path: str = None, npz_path: str = None):
    output_folder = output_folder / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    detector = GestureDetector(model_type=model_name)
    if video_path:
        test_video(video_path, output_folder, detector, name=model_name)
    if npz_path:
        test_npz(npz_path, output_folder, detector, name=model_name)

def test_realtime_gesture_detector():
    detector = RealtimeGestureDetector(
        confidence_threshold=0.2,
        min_gap_s=0.2,
        min_length_s=0.3,
    )

    raw_results, segments = detector.process_webcam(
        duration=30,
        camera_index=1,
        show_display=True,
        save_video=True,
        apply_post_processing=True,
    )


if __name__ == "__main__":
    main()