import os
from extract_clips_gesres import extract_clips_gesres
from extract_clips_multisimo import extract_clips_multisimo
from extract_clips_saga import extract_clips_saga
from extract_clips_saga_plus import extract_clips_saga_plus
from extract_clips_zhubo import extract_clips_zhubo
from extract_clips_ecolang import extract_clips_ecolang
from extract_clips_tedm3d import extract_clips_tedm3d
from utils import check_ffmpeg

if __name__ == "__main__":
    # Change labels as needed
    gesture_label = "Gesture"
    no_gesture_label = "NoGesture"
    move_label = "Move"

    # Update corpus directory paths as needed
    ecolang_dir = "Ecolang/trainingraw/"
    multisimo_dir = "Multisimo/videos_annotations/"
    saga_dir = "SAGA/"
    saga_plus_dir = "SAGAplus"
    tedm3d_dir = "TedM3D/processed/"
    gesres_dir = "GESRES/01Gesture_videos/"
    zhubo_dir = "ZHUBO/zhubo_split_9"

    # Change output directory path as needed
    output_dir = "CorpusClips"
    os.makedirs(output_dir, exist_ok=True)
    
    gesture_dir = os.path.join(output_dir, gesture_label)
    no_gesture_dir = os.path.join(output_dir, no_gesture_label)
    move_dir = os.path.join(output_dir, move_label)
    os.makedirs(gesture_dir, exist_ok=True)
    os.makedirs(no_gesture_dir, exist_ok=True)
    os.makedirs(move_dir, exist_ok=True)

    vars = dict(
        gesture_label=gesture_label,
        no_gesture_label=no_gesture_label,
        move_label=move_label,
        gesture_dir=gesture_dir,
        no_gesture_dir=no_gesture_dir,
        move_dir=move_dir
    )

    if not check_ffmpeg():
        print("Error: ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH.")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg (Ubuntu/Debian)")
        exit(1)

    extract_clips_ecolang(ecolang_dir, vars)
    extract_clips_multisimo(multisimo_dir, vars)
    extract_clips_saga(saga_dir, vars)
    extract_clips_saga_plus(saga_plus_dir, vars)
    extract_clips_tedm3d(tedm3d_dir, vars)
    extract_clips_gesres(gesres_dir, vars)
    extract_clips_zhubo(zhubo_dir, vars)
