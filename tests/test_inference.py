# from parent folder: python -m tests.test_inference
import numpy as np
from pathlib import Path

from envisionhgdetector import GestureDetector, CombinedGestureDetector, RealtimeGestureDetector
from envisionhgdetector import label_video

def main():
    # ------------------- Test Video -------------------
    video_folder = Path("path/to/video_folder")  # Replace with the desired video folder path
    if not video_folder.exists():
        raise FileNotFoundError(f"Video folder {video_folder} does not exist.")

    output_folder = Path("path/to/output_folder")  # Replace with the desired output folder path
    output_folder.mkdir(parents=True, exist_ok=True)

    video_paths = list(video_folder.glob("*.mp4")) # find all the videos in folder
    for video_path in video_paths:
        test_gesture_detector(output_folder, model_name="cnn_b", video_path=video_path)
        test_gesture_detector(output_folder, model_name="lightgbm", video_path=video_path)
        test_combined_gesture_detector(output_folder, video_path=video_path)

    # ------------------- Test NPZ -------------------
    npz_path = Path("path/to/data.npz")
    if npz_path is not None and not npz_path.exists():
        raise FileNotFoundError(f"NPZ file {npz_path} does not exist.")
    test_gesture_detector(output_folder, model_name="cnn_b", npz_path=npz_path)
    test_gesture_detector(output_folder, model_name="lightgbm", npz_path=npz_path)
    test_combined_gesture_detector(output_folder, npz_path=npz_path)

    # ------------------- Test Realtime -------------------
    test_realtime_gesture_detector()


def test_video(video_path: str, output_folder: str, detector, name:str):
    print(f"testing video: {video_path} with detector: {name} saving results to: {output_folder}")
    result = detector.process_video(
        video_path,
        output_folder,
    ) 
    label_video(
        video_path=video_path,
        output_folder=str(output_folder),
    )

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
    results_df.to_csv(output_folder / npz_path.stem / f"{npz_path.stem}_predictions.csv", index=False)

def test_combined_gesture_detector(output_folder: str, video_path: str = None, npz_path: str = None):
    output_folder = output_folder / "combined_results"
    output_folder_video = output_folder / "video"
    output_folder_npz = output_folder / "npz"

    output_folder_video.mkdir(parents=True, exist_ok=True)
    output_folder_npz.mkdir(parents=True, exist_ok=True)

    detector = CombinedGestureDetector()
    if video_path:
        test_video(video_path, output_folder_video, detector, name="Combined")
    if npz_path:
        test_npz(npz_path, output_folder_npz, detector, name="Combined")

def test_gesture_detector(output_folder: str, model_name: str, video_path: str = None, npz_path: str = None):
    output_folder = output_folder / model_name
    output_folder_video = output_folder / "video"
    output_folder_npz = output_folder / "npz"

    output_folder_video.mkdir(parents=True, exist_ok=True)
    output_folder_npz.mkdir(parents=True, exist_ok=True)

    # running on default thresholds
    detector = GestureDetector(model_type=model_name)
    if video_path:
        test_video(video_path, output_folder_video, detector, name=model_name)
    if npz_path:
        test_npz(npz_path, output_folder_npz, detector, name=model_name)

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