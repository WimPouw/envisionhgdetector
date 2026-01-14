"""
Test script for evaluating the trained LightGBM gesture model on video files.

Usage:
    python test_model.py --video ../../sample_data/tedkid_sample.mp4
    python test_model.py --video ../../sample_data/tyson_sample.mp4 --output results.json
"""

import argparse
import sys
import json
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

try:
    # Import directly to avoid tensorflow dependency from model_cnn
    from envisionhgdetector.model_lightgbm import LightGBMGestureModel
    from envisionhgdetector.config import Config
except ImportError as e:
    print(f"Error importing: {e}")
    print("Make sure you're in the correct environment with required packages.")
    print("Required: opencv-python, mediapipe, lightgbm, joblib, scikit-learn")
    sys.exit(1)


def test_model_on_video(video_path: str, model: LightGBMGestureModel,
                        show_preview: bool = False) -> dict:
    """
    Run the model on a video file and collect predictions.

    Args:
        video_path: Path to video file
        model: Loaded LightGBMGestureModel instance
        show_preview: Show video preview with predictions

    Returns:
        Dictionary with test results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {Path(video_path).name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")

    predictions = []
    confidences = []
    frame_times = []

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            # Extract features from frame
            features = model.extract_features_from_frame(frame)

            if features is not None:
                # Run prediction
                probs = model.predict(features)
                pred_idx = np.argmax(probs[0])
                pred_label = model.label_encoder.inverse_transform([pred_idx])[0]
                confidence = probs[0][pred_idx]

                predictions.append(pred_label)
                confidences.append(float(confidence))
            else:
                # Not enough frames in buffer yet
                predictions.append("BUFFERING")
                confidences.append(0.0)

            frame_times.append(time.time() - frame_start)
            frame_count += 1

            # Show preview if requested
            if show_preview and features is not None:
                # Draw prediction on frame
                label = predictions[-1]
                conf = confidences[-1]
                color = (0, 255, 0) if label == "Gesture" else (0, 0, 255)
                cv2.putText(frame, f"{label}: {conf:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.imshow("Model Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress update every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")

    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time

    # Filter out buffering frames for statistics
    valid_predictions = [p for p in predictions if p != "BUFFERING"]
    valid_confidences = [c for i, c in enumerate(confidences) if predictions[i] != "BUFFERING"]

    # Count predictions
    pred_counts = Counter(valid_predictions)

    # Calculate statistics
    results = {
        "video_path": str(video_path),
        "video_name": Path(video_path).name,
        "timestamp": datetime.now().isoformat(),
        "video_info": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": total_frames / fps
        },
        "model_info": {
            "gesture_labels": model.gesture_labels,
            "window_size": model.window_size,
            "includes_fingers": model.includes_fingers,
            "includes_dynamics": model.includes_dynamics,
            "expected_features": model.expected_features
        },
        "results": {
            "total_predictions": len(valid_predictions),
            "prediction_counts": dict(pred_counts),
            "prediction_percentages": {
                k: f"{v/len(valid_predictions)*100:.1f}%"
                for k, v in pred_counts.items()
            },
            "mean_confidence": float(np.mean(valid_confidences)) if valid_confidences else 0,
            "min_confidence": float(np.min(valid_confidences)) if valid_confidences else 0,
            "max_confidence": float(np.max(valid_confidences)) if valid_confidences else 0
        },
        "performance": {
            "total_time_seconds": total_time,
            "avg_frame_time_ms": float(np.mean(frame_times)) * 1000,
            "fps_achieved": frame_count / total_time,
            "realtime_capable": (np.mean(frame_times) * 1000) < (1000 / fps)
        }
    }

    return results


def print_results(results: dict):
    """Print results in a formatted way."""
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    print(f"\nVideo: {results['video_name']}")
    print(f"Duration: {results['video_info']['duration_seconds']:.1f}s at {results['video_info']['fps']:.1f} FPS")

    print(f"\nModel Configuration:")
    print(f"  Window size: {results['model_info']['window_size']} frames")
    print(f"  Features: {results['model_info']['expected_features']}")
    print(f"  Finger landmarks: {'Yes' if results['model_info']['includes_fingers'] else 'No'}")
    print(f"  Motion dynamics: {'Yes' if results['model_info']['includes_dynamics'] else 'No'}")

    print(f"\nPrediction Summary:")
    for label, count in results['results']['prediction_counts'].items():
        pct = results['results']['prediction_percentages'][label]
        print(f"  {label}: {count} frames ({pct})")

    print(f"\nConfidence:")
    print(f"  Mean: {results['results']['mean_confidence']:.3f}")
    print(f"  Range: {results['results']['min_confidence']:.3f} - {results['results']['max_confidence']:.3f}")

    print(f"\nPerformance:")
    print(f"  Total time: {results['performance']['total_time_seconds']:.1f}s")
    print(f"  Avg frame time: {results['performance']['avg_frame_time_ms']:.1f}ms")
    print(f"  Achieved FPS: {results['performance']['fps_achieved']:.1f}")
    print(f"  Real-time capable: {'Yes' if results['performance']['realtime_capable'] else 'No'}")


def main():
    parser = argparse.ArgumentParser(description='Test LightGBM gesture model on video')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview with predictions')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (optional, uses default if not specified)')

    args = parser.parse_args()

    # Resolve video path
    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = Path(__file__).parent / args.video

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print("Loading model...")

    # Load model
    if args.model:
        config = Config()
        config.weights_path = args.model
        model = LightGBMGestureModel(config)
    else:
        model = LightGBMGestureModel()

    print(f"Model loaded successfully!")
    print(f"  Gestures: {model.gesture_labels}")

    # Run test
    print("\nRunning inference on video...")
    results = test_model_on_video(str(video_path), model, show_preview=args.preview)

    # Print results
    print_results(results)

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    # Cleanup
    model.close()


if __name__ == '__main__':
    main()
