# EnvisionHGDetector

Gesture detection package supporting CNN-B, LightGBM, realtime webcam detection, and combined CNN + LightGBM inference.

This guide covers installation, CLI usage, and the Python API for the current package layout.

## Install for use or development

### Use the published package

Use Python 3.10 and install the published package into your environment:

```bash
pip install envisionhgdetector
```

### Develop from source

From the repository root, create a Conda environment with Python 3.10 and install the local package in editable mode:

```bash
conda create -n envision-dev python=3.10 pip
conda activate envision-dev
pip install -e .
```

The editable install uses `setup.py` and `requirements.txt` for dependencies. Edits to the Python source are then available in the environment without reinstalling the package.

The package uses CPU TensorFlow and depends on MediaPipe, OpenCV, LightGBM, NumPy, and pandas.

## Quick Import Test

```powershell
python -c "import envisionhgdetector; print('import ok')"
```

If this fails with a missing package such as `tensorflow` or `mediapipe`, activate the correct environment or install the missing dependency.

## Command-Line Interface

The CLI entry point is:

```powershell
envisionhgdetector.use_cli --help
```

### Realtime webcam detector

Run the LightGBM realtime detector with the default camera:

```powershell
envisionhgdetector.use_cli `
  --detector realtime `
  --confidence-threshold 0.2 `
  --min-gap 0.2 `
  --min-length 0.3 `
  --camera-index 0
```

Useful realtime options:

- `--duration 30` limits the session to 30 seconds.
- `--no-display` disables the OpenCV preview window.
- `--no-save-video` avoids saving the annotated webcam video.
- `--no-post-processing` skips segment refinement.
- Press `Q` in the preview window to stop an unlimited session.

Realtime output is saved under:

```text
output_realtime/session_YYYYMMDD_HHMMSS/
```

Typical files include:

- `raw_frame_results.csv`
- `gesture_segments.csv`
- `gesture_segments.eaf` when an annotated video is saved
- `webcam_session.mp4` when video saving is enabled
- session summary CSV files

### Installation test

The test initializes the realtime detector without opening the webcam:

```powershell
python envisionhgdetector.use_cli --test
```

### Analyze a saved realtime session

Analyze the newest session:

```powershell
python envisionhgdetector.use_cli --analyze-session
```

Analyze a specific session:

```powershell
python envisionhgdetector.use_cli `
  --analyze-session output_realtime\session_20260923_120000
```

### Default video detector

The `default` detector processes a video file using the selected model.

CNN-B is the currently supported CNN option:

```powershell
python envisionhgdetector.use_cli `
  --detector default `
  --model cnn_b `
  --video path\to\input.mp4 `
  --output-folder output_cnn_b
```

LightGBM video processing:

```powershell
python envisionhgdetector.use_cli `
  --detector default `
  --model lightgbm `
  --video path\to\input.mp4 `
  --output-folder output_lightgbm
```

Common options:

- `--stride 2` samples every second frame where supported.
- `--elan-only` writes the ELAN output without saving the other prediction artifacts.
- `--config path\to\config.yaml` supplies a custom model configuration.
- `--weights path\to\weights.h5` or `.pkl` supplies custom model weights.

Default video output normally includes predictions, segments, features, a labeled video, and an ELAN file.

## Combined Detector

The combined detector is available as a Python class. The current `use_cli.py` branch for `--detector combined` is intentionally not implemented yet.

```python
from envisionhgdetector import CombinedGestureDetector


detector = CombinedGestureDetector(
    cnn_config_path=None,
    lightgbm_config_path=None,
    cnn_weights_path=None,
    lightgbm_weights_path=None,
)

result = detector.process_video(
    video_path=r"path\to\input.mp4",
    output_folder=r"output_combined",
    stride=1,
)

print(result)
```

The combined detector:

- Runs CNN and LightGBM predictions.
- Aligns model results by `frame_index`.
- Fuses gesture, movement, and no-gesture confidences.
- Creates combined segments.
- Saves prediction and segment CSV files.
- Saves model features when available.
- Creates a labeled video and ELAN annotation.

For pre-extracted world landmarks:

```python
import numpy as np
from envisionhgdetector import CombinedGestureDetector


detector = CombinedGestureDetector()
landmarks = np.load(r"path\to\landmarks.npz")

combined = detector.predict_labels_from_landmarks(
    landmarks_per_frame=landmarks,
    fps=25.0,
    stride=1,
)

combined.to_csv(r"output_combined\landmark_predictions.csv", index=False)
```

The landmark array is expected to have shape:

```text
(number_of_frames, 92)
```

## Direct Python API

### Single model

```python
from envisionhgdetector import GestureDetector


detector = GestureDetector(model_type="cnn_b")
predictions, stats, segments, features, timestamps = detector.predict_video(
    r"path\to\input.mp4",
    stride=1,
)
```

Use `model_type="lightgbm"` for LightGBM. The current `cnn` configuration is reserved for a future implementation and may raise `NotImplementedError`.

### Realtime API

```python
from envisionhgdetector import RealtimeGestureDetector


detector = RealtimeGestureDetector(
    confidence_threshold=0.2,
    min_gap_s=0.2,
    min_length_s=0.3,
)

raw_results, segments = detector.process_webcam(
    duration=30,
    camera_index=0,
    show_display=True,
    save_video=True,
    apply_post_processing=True,
)
```

## Advanced Processing

After `GestureDetector.process_video()` or `process_folder()` has saved predictions and a labeled video, you can cut detected gestures into clips, retrack their world landmarks, and calculate kinematic features and DTW distances:

```python
from pathlib import Path

from envisionhgdetector import GestureDetector, utils

output_folder = Path("output_detection")
detector = GestureDetector(model_type="cnn_b")

clips = utils.cut_video_by_segments(str(output_folder))
tracking = detector.retrack_gestures(
    input_folder=str(output_folder / "gesture_segments"),
    output_folder=str(output_folder / "retracked"),
)
analysis = detector.analyze_dtw_kinematics(
    landmarks_folder=tracking["landmarks_folder"],
    output_folder=str(output_folder / "analysis"),
)
```

The `clips` dictionary maps source videos to generated clip paths. `tracking` and `analysis` contain paths to their generated files, or an `error` key if a step fails.

## Output Files

`GestureDetector.process_video()` writes files directly to its `output_folder`. With a video named `example.mp4`, it produces:

| File | Contents |
| --- | --- |
| `example.mp4_predictions.csv` | Frame predictions, timestamps, and confidence values |
| `example.mp4_segments.csv` | Detected gesture intervals |
| `example.mp4_features.npy` | Extracted features, when available |
| `labeled_example.mp4` | Video with prediction overlays |
| `example.mp4.eaf` | ELAN annotation |

With `elan_only=True`, the detector writes the `.eaf` file without the prediction, segment, feature, or labeled video files. The combined detector uses the same main filenames and may also write separate `*_cnn_features.npy` and `*_lightgbm_features.npy` files.

The advanced processing steps create `gesture_segments/` with gesture clips and feature arrays, `retracked/` with tracked videos and `*_world_landmarks.npy` and `*_visibility.npy` arrays, and `analysis/` with `dtw_distances.csv`, `kinematic_features.csv`, and `gesture_visualization.csv`.

`RealtimeGestureDetector.process_webcam()` writes a timestamped `output_realtime/session_YYYYMMDD_HHMMSS/` directory. It contains `raw_frame_results.csv` and `session_summary.csv`; when segments are found, it also contains `gesture_segments.csv` and `session_summary_detailed.csv`. Saving video produces `webcam_session.mp4`, and an ELAN file is created when saved video and segments are available.

## Shared Output Conventions

Internal prediction DataFrames use:

- `frame_index`
- `prediction`
- `confidence`
- `motion_confidence`
- `gesture_confidence`
- `no_gesture_confidence`
- `move_confidence`
- `timestamp`

Labels are defined in `state.py` as:

- `Gesture`
- `NoGesture`
- `Move`

Segment processing supports explicit Move behavior through `MoveMode`:

```python
from envisionhgdetector.state import MoveMode

mode = MoveMode.SEPARATE       # Keep Move as a separate label
mode = MoveMode.AS_GESTURE     # Treat Move as Gesture
mode = MoveMode.IGNORE         # Treat Move as NoGesture
```

## Troubleshooting

### Missing TensorFlow or MediaPipe

Activate the environment used by the project, then rerun the import test. The package imports CNN and realtime components during initialization, so missing model dependencies can prevent even unrelated commands from importing.

### Missing model files

When no custom paths are provided, `DefaultConfig` looks for model files under the package's configured model directories. Use `--config` and `--weights` when testing with local model artifacts.

### No webcam

Use `--camera-index 1` or another available device index. For automated or headless testing, use `--test` instead of starting webcam processing.

<!-- ## Current Limitations

- The CLI combined-detector branch is not implemented; use the Python API shown above.
- The `cnn` model configuration is not currently enabled by `DefaultConfig`; use `cnn_b` or `lightgbm`.
- Full runtime testing requires the project's machine-learning and video dependencies. -->
