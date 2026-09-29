"""Command-line entry point for Envision gesture detector demos."""

'''
Incomplete:
- idk how to handle combined detection with multiple models. currently it is hardcoded
'''

import argparse
import glob
import os
import sys

from envisionhgdetector.state import ModelNames, Thresholds, VALID_MODEL_NAMES, Labels
from envisionhgdetector.utils import valid_float
from envisionhgdetector import (
    CombinedGestureDetector,
    GestureDetector,
    RealtimeGestureDetector,
)

def validate_floats_args(args: argparse.Namespace):
    if not valid_float(args.confidence_threshold):
        raise argparse.ArgumentTypeError("Confidence threshold must be a float between 0 and 1.")
    if not valid_float(args.min_gap):
        raise argparse.ArgumentTypeError("Minimum gap must be a float between 0 and 1.")
    if not valid_float(args.min_length):
        raise argparse.ArgumentTypeError("Minimum length must be a float between 0 and 1.")

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for all detector modes."""
    parser = argparse.ArgumentParser(
        description="Run Envision gesture detection with a webcam or video file."
    )
    parser.add_argument(
        "--detector",
        choices=("realtime", "default", "combined"),
        default="realtime",
        help="Detector to run (default: realtime).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Load the realtime detector without opening a webcam.",
    )
    parser.add_argument(
        "--analyze-session",
        nargs="?",
        const="latest",
        metavar="FOLDER",
        help="Analyze a saved realtime session, or the latest session if omitted.",
    )
    parser.add_argument("--output-folder", default="output_detection")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--min-gap", type=float, default=0.2)
    parser.add_argument("--min-length", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--elan-only", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--no-post-processing", action="store_true")

    # REALTIME DETECTOR ARGS
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--duration", type=float)

    # DEFAULT SPECIFIC DETECTOR ARGS
    parser.add_argument("--video", help="Input video for default or combined detection.")
    parser.add_argument(
        "--model",
        choices=VALID_MODEL_NAMES,
        default=ModelNames.CNN,
        help="Model used by the default detector (default: cnn).",
    )
    parser.add_argument("--config")
    parser.add_argument("--weights")

    return parser

def quick_test() -> bool:
    """Load the realtime detector to verify the installation."""
    try:
        default_threshold = 0.2
        detector = RealtimeGestureDetector(default_threshold, default_threshold, default_threshold)
        print("Realtime detector initialized successfully.")
        print(f"Model features: {detector.model.expected_features}")
        print(f"Gesture labels: {detector.model.gesture_labels}")
        return True
    except Exception as error:
        print(f"Installation test failed: {error}", file=sys.stderr)
        return False

def analyze_session(session_folder: str | None):
    """Analyze a saved realtime session."""
    if session_folder in (None, "latest"):
        session_folders = glob.glob("output_realtime/session_*")
        if not session_folders:
            raise FileNotFoundError("No sessions found in output_realtime/")
        session_folder = max(session_folders, key=os.path.getctime)

    default_threshold = 0.2
    detector = RealtimeGestureDetector(default_threshold, default_threshold, default_threshold)
    raw_df, segments_df = detector.load_and_analyze_session(session_folder)
    print(f"Session: {session_folder}")
    print(f"Frames: {len(raw_df)}")
    if not raw_df.empty:
        gesture_count = (raw_df[Labels.GESTURE] != Labels.NOGESTURE).sum()
        print(f"{Labels.GESTURE} frames: {gesture_count} ({gesture_count / len(raw_df) * 100:.1f}%)")
    print(f"Segments: {len(segments_df)}")
    return raw_df, segments_df

def run_from_arguments(args: argparse.Namespace):
    """Instantiate the selected detector and run its matching workflow."""

    if args.test:
        return quick_test()
    if args.analyze_session is not None:
        return analyze_session(args.analyze_session)

    validate_floats_args(args)
    thresholds = Thresholds(
        motion_threshold=args.confidence_threshold,
        gesture_threshold=args.confidence_threshold,
        min_gap_s=args.min_gap,
        min_length_s=args.min_length,
    )

    if args.detector == "realtime":
        detector = RealtimeGestureDetector(
            confidence_threshold=args.confidence_threshold,
            min_gap_s=args.min_gap,
            min_length_s=args.min_length,
        )
        return detector.process_webcam(
            duration=args.duration,
            camera_index=args.camera_index,
            show_display=not args.no_display,
            save_video=not args.no_save_video,
            apply_post_processing=not args.no_post_processing,
        )
    elif args.detector == "default":
        if not args.video:
            raise ValueError("--video is required for default and combined detectors.")
        detector = GestureDetector(
            model_type=args.model,
            config_path=args.config,
            weights_path=args.weights,
            thresholds=thresholds,
        )
        return detector.process_video(
            args.video,
            args.output_folder,
            elan_only=args.elan_only,
        )

    elif args.detector == "combined":
        raise NotImplementedError("Combined detector is not yet implemented.")
    else:
        raise ValueError(f"Unknown detector: {args.detector}")

def main() -> int:
    """Parse arguments and run the selected workflow."""
    args = build_argument_parser().parse_args()
    try:
        run_from_arguments(args)
    except KeyboardInterrupt:
        print("\nSession interrupted by user")
        return 130
    except Exception as error:
        raise ValueError(f"Error: {error}")

if __name__ == "__main__":
    raise SystemExit(main())
