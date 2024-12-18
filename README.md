# EnvisionHGDetector: Hand Gesture Detection Using Convolutional Neural Networks

A Python package for detecting and classifying hand gestures using MediaPipe Holistic and deep learning.

<div align="center">Wim Pouw (wim.pouw@donders.ru.nl)</div>

## Info
This package provides a straightforward way to detect hand gestures in videos using a combination of MediaPipe Holistic features and a convolutional neural network (CNN). The package performs:

* Feature extraction using MediaPipe Holistic (hand, body, and face features)
* Real-time gesture detection using a pre-trained CNN model, that I trained on SAGA, TEDM3D dataset, and the zhubo open gesture annotated datasets.
* Automatic annotation of videos with gesture classifications
* Output generation in CSV format and ELAN-compatible files, and video labeled

Currently, the detector can identify:
- General motion states
- Specific hand gestures ("Gesture")
- Movement patterns ("Move"; this is only trained on SAGA, because these also annotated movements that were not gestures, like nose scratching); it will therefore be an unreliably category perhaps

## Installation

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
    gesture_threshold=0.8,   # Confidence threshold for gestures
    min_gap_s=0.3,          # Minimum gap between gestures
    min_length_s=0.3        # Minimum gesture duration
)

# Process videos
results = detector.process_folder(
    video_folder="path/to/videos",
    output_folder="path/to/output"
)
```

## Features

The detector uses 29 features extracted from MediaPipe Holistic, including:
- Head rotations
- Hand positions and movements
- Body landmark distances
- Normalized motion metrics

## Output

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

Pouw, W. (2024). EnvisionHGDetector: Hand Gesture Detection Using Convolutional Neural Networks [Version 0.0.1]. Retrieved from: [repository URL]

### Additional Citations

MediaPipe:
Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. arXiv preprint arXiv:1906.08172.

## Contributing

Feel free to help improve this code. As this is primarily aimed at making gesture detection accessible for research purposes, contributions focusing on usability and reliability are especially welcome.

## Support

For issues and questions, please use the GitHub issue tracker. For academic inquiries, contact [your contact info].
