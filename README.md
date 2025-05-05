# EnvisionHGDetector: Hand Gesture Detection Using Convolutional Neural Networks
A Python package for detecting and classifying hand gestures using MediaPipe Holistic and deep learning.
<div align="center">Wim Pouw (wim.pouw@donders.ru.nl), Bosco Yung, Sharjeel Shaikh, James Trujillo, Gerard de Melo, Babajide Owoyele</div>

<div align="center">
<img src="images/ex.gif" alt="Hand Gesture Detection Demo">
</div>

## Info
This package provides a straightforward way to detect hand gestures in a variety of videos using a combination of MediaPipe Holistic features and a convolutional neural network (CNN). We plan to update this package with better predicting network in the near future, and we plan to also make an evaluation report so that it is clear how it performs for several types of videos. For now, feel free to experiment. If your looking to just quickly generate isolate some gestures into elan, this is the package for you. Do note that annotation by rates will be much superior to this gesture coder.

The package performs:

* Feature extraction using MediaPipe Holistic (hand, body, and face features)
* Post-hoc gesture detection using a pre-trained CNN model, that I trained on SAGA, TEDM3D dataset, and the zhubo, open gesture annotated datasets.
* Automatic annotation of videos with gesture classifications
* Output generation in CSV format and ELAN-compatible files, and video labeled

Currently, the detector can identify:
- Just a general hand gesture, ("Gesture")
- Movement patterns ("Move"; this is only trained on SAGA, because these also annotated movements that were not gestures, like nose scratching); it will therefore be an unreliable category perhaps

## Installation
Consider creating a conda environment first (conda create -n envision python==3.9; conda activate envision).
```bash
conda create -n envision python==3.9
conda activate envision
(envision) pip install envisionhgdetector
```
otherwise install like this
```bash
pip install envisionhgdetector
```

Note: This package is CPU-only for wider compatibility and ease of use.

## Quick Start

```python
from envisionhgdetector import GestureDetector

# Initialize detector
detector = GestureDetector(
    motion_threshold=0.8,    # Sensitivity to motion
    gesture_threshold=0.5,   # Confidence threshold for gestures
    min_gap_s=0.1,           # Minimum gap between gestures
    min_length_s=0.1,        # Minimum gesture duration
    gesture_class_bias=0.0   # How much do you want to increase likelihood of classification to gesture (versus self-adaptor "move" label)
)

# Process videos
results = detector.process_folder(
    input_folder="path/to/videos",
    output_folder="path/to/output"
)

# additional processing of videos to segmented videos cut by gesture event
from envisionhgdetector import utils
segments = utils.cut_video_by_segments(outputfolder)

# retrack the videos with metric pose world landmarks
# Step 3: Create paths
gesture_segments_folder = os.path.join(outputfolder, "gesture_segments")
retracked_folder = os.path.join(outputfolder, "retracked")
analysis_folder = os.path.join(outputfolder, "analysis")

print(f"\nLooking for segments in: {gesture_segments_folder}")
if os.path.exists(gesture_segments_folder):
    segment_files = [f for f in os.listdir(gesture_segments_folder) if f.endswith('.mp4')]
    print(f"Found {len(segment_files)} segment files")
else:
    print("Gesture segments folder not found!")

# Step 3: Retrack gestures with world landmarks
print("\nStep 4: Retracking gestures...")
tracking_results = detector.retrack_gestures(
    input_folder=gesture_segments_folder,
    output_folder=retracked_folder
)
print(f"Tracking results: {tracking_results}")

# Compute DTW distance matrix and kinematic features
analysis_results = detector.analyze_dtw_kinematics(
        landmarks_folder=tracking_results["landmarks_folder"],
        output_folder=analysis_folder
    )
print(f"Analysis results: {analysis_results}")

# Create dashboard (then launched in python, python app.py [in a seperate teriminal])
detector.prepare_gesture_dashboard(
    data_folder=analysis_folder
    )
```

## Features

The detector uses 29 features extracted from MediaPipe Holistic, including:
- Head rotations
- Hand positions and movements
- Body landmark distances
- Normalized feature metrics

## Output

The detector generates three types of output in your specified output folder:

1. Automated Annotations (`/output/automated_annotations/`)
   - CSV files with frame-by-frame predictions
   - Contains confidence values and classifications for each frame
   - Format: `video_name_confidence_timeseries.csv`

2. ELAN Files (`/output/elan_files/`)
   - ELAN-compatible annotation files (.eaf)
   - Contains time-aligned gesture segments
   - Useful for manual verification and research purposes
   - Format: `video_name.eaf`

3. Labeled Videos (`/output/labeled_videos/`)
   - Processed videos with visual annotations
   - Shows real-time gesture detection and confidence scores
   - Useful for quick verification of detection quality
   - Format: `labeled_video_name.mp4`

4. Retracked Videos (`/output/retracked/`)
   - rendered tracked videos and pose world landmarks

5. Kinematic analysis (`output/analyis/`)
   - DTW distance matrix (.csv) between all gesture comparisons
   - Kinematic features (.csv) per gesture (e.g., number of submovements, max speed, max acceleration)
   - Gesture visualization (.csv; UMAP of DTW distance matrix, for input for Dashboard)

6. Dashboard (`/output/app.py`)
   - This app visualizes the gesture similarity space and shows the kinematic features, the user can click on the videos and identify metrics

## Technical Background

The package builds on previous work in gesture detection, particularly focused on using MediaPipe Holistic for comprehensive feature extraction. The CNN model is designed to handle complex temporal patterns in the extracted features.

## Requirements
- Python 3.7+
- tensorflow-cpu
- mediapipe
- opencv-python
- numpy
- pandas

## Citation

If you use this package, please cite:

Pouw, W., Yung, B., Shaikh, S., Trujillo, J., Rueda-Toicen, A., de Melo, G., Owoyele, B. (2024). envisionhgdetector: Hand Gesture Detection Using a Convolutional Neural Network (Version 0.0.5.0) [Computer software]. https://pypi.org/project/envisionhgdetector/

### Additional Citations

Zhubo dataset (used for training):
* Bao, Y., Weng, D., & Gao, N. (2024). Editable Co-Speech Gesture Synthesis Enhanced with Individual Representative Gestures. Electronics, 13(16), 3315.

SAGA dataset (used for training)
* Lücking, A., Bergmann, K., Hahn, F., Kopp, S., & Rieser, H. (2010). The Bielefeld speech and gesture alignment corpus (SaGA). In LREC 2010 workshop: Multimodal corpora–advances in capturing, coding and analyzing multimodality.

TED M3D:
* Rohrer, Patrick. A temporal and pragmatic analysis of gesture-speech association: A corpus-based approach using the novel MultiModal MultiDimensional (M3D) labeling system. Diss. Nantes Université; Universitat Pompeu Fabra (Barcelone, Espagne), 2022.

MediaPipe:
* Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172.

Adapted CNN Training and inference code:
* Pouw, W. (2024). EnvisionBOX modules for social signal processing (Version 1.0.0) [Computer software]. https://github.com/WimPouw/envisionBOX_modulesWP

Original Noddingpigeon Training code:
* Yung, B. (2022). Nodding Pigeon (Version 0.6.0) [Computer software]. https://github.com/bhky/nodding-pigeon

Some code I reused for creating ELAN files came from Cravotta et al., 2022:
* Ienaga, N., Cravotta, A., Terayama, K., Scotney, B. W., Saito, H., & Busa, M. G. (2022). Semi-automation of gesture annotation by machine learning and human collaboration. Language Resources and Evaluation, 56(3), 673-700.

## Contributing
Feel free to help improve this code. As this is primarily aimed at making automatic gesture detection easily accessible for research purposes, contributions focusing on usability and reliability are especially welcome (happy to collaborate, just reach out to wim.pouw@donders.ru.nl).

