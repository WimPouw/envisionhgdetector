import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from envisionhgdetector import GestureDetector
from .state import Thresholds
from .utils import create_elan_file, create_segments, get_video_fps, label_video

class CombinedGestureDetector:
    """
    A combined gesture detector that merges predictions from CNN and LightGBM models.

    NOTE: Post-processing wrappers currently delegate through the CNN detector
    and remain CNN-focused in naming/configuration.
    """
    def __init__(self,
        cnn_weight: float = 0.5,
        lgbm_weight: float = 0.5,      
        cnn_config_path: Optional[Path] = None,
        lightgbm_config_path: Optional[Path] = None,
        cnn_weights_path: Optional[Path] = None,
        lightgbm_weights_path: Optional[Path] = None,
        cnn_thresholds: Optional[Thresholds] = None,
        lightgbm_thresholds: Optional[Thresholds] = None
        ):
        self.cnn_detector = GestureDetector(model_type="cnn", config_path=cnn_config_path, weights_path=cnn_weights_path, thresholds=cnn_thresholds)
        self.lgbm_detector = GestureDetector(model_type="lightgbm", config_path=lightgbm_config_path, weights_path=lightgbm_weights_path, thresholds=lightgbm_thresholds)
        self.cnn_weight = cnn_weight
        self.lgbm_weight = lgbm_weight

    def predict_video(self, video_path: str, stride: int = 1):
        """
        Predict gestures in a video using both CNN and LightGBM models and combine the results.

        Args:
            video_path: Path to the input video file.

        Returns:
            Tuple of combined results, statistics, segments, raw predictions,
            and timestamps. The first two model result tuples are retained on
            the instance for comparison.
        """
        cnn_output = self.cnn_detector.predict_video(video_path, stride)
        lgbm_output = self.lgbm_detector.predict_video(video_path, stride)

        cnn_results, cnn_stats, _, cnn_features, cnn_timestamps = cnn_output
        lgbm_results, lgbm_stats, _, lgbm_features, lgbm_timestamps = lgbm_output

        combined_results = self.combine_frame_predictions(
            cnn_results=cnn_results,
            lgbm_results=lgbm_results,
            cnn_weight=self.cnn_weight,
            lgbm_weight=self.lgbm_weight
        )

        segment_input = combined_results.copy()
        segment_input["time"] = segment_input["timestamp"]
        segments = create_segments(
            segment_input,
            label_column="prediction",
            min_gap_s=max(
                self.cnn_detector.config.thresholds.min_gap_s,
                self.lgbm_detector.config.thresholds.min_gap_s,
            ),
            min_length_s=max(
                self.cnn_detector.config.thresholds.min_length_s,
                self.lgbm_detector.config.thresholds.min_length_s,
            ),
        )

        stats = {
            "model_type": "combined",
            "cnn_stats": cnn_stats,
            "lightgbm_stats": lgbm_stats,
            "average_motion": float(combined_results["motion_confidence"].mean()),
            "average_gesture": float(combined_results["gesture_confidence"].mean()),
            "average_move": float(combined_results["move_confidence"].mean()),
            "cnn_weight": self.cnn_weight,
            "lgbm_weight": self.lgbm_weight,
        }

        raw_predictions = combined_results[[
            "motion_confidence",
            "gesture_confidence",
            "move_confidence",
        ]].to_numpy()
        timestamps = combined_results["timestamp"].tolist()

        self.last_cnn_output = cnn_output
        self.last_lgbm_output = lgbm_output
        self.last_cnn_features = cnn_features
        self.last_lgbm_features = lgbm_features
        self.last_cnn_timestamps = cnn_timestamps
        self.last_lgbm_timestamps = lgbm_timestamps

        return combined_results, stats, segments, raw_predictions, timestamps

    def process_video(
        self,
        video_path: str,
        output_folder: str,
        elan_only: bool = False,
        stride: int = 1,
    ) -> Dict[str, object]:
        """Run both models and save combined video outputs."""
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        os.makedirs(output_folder, exist_ok=True)
        video_name = os.path.basename(video_path)

        try:
            predictions, stats, segments, _, timestamps = self.predict_video(
                video_path,
                stride=stride,
            )
            if predictions.empty:
                return {"error": "No predictions generated"}

            if not elan_only:
                predictions_path = os.path.join(
                    output_folder, f"{video_name}_predictions.csv"
                )
                segments_path = os.path.join(
                    output_folder, f"{video_name}_segments.csv"
                )
                predictions.to_csv(predictions_path, index=False)
                segments.to_csv(segments_path, index=False)

                if hasattr(self, "last_cnn_features") and len(self.last_cnn_features) > 0:
                    np.save(
                        os.path.join(output_folder, f"{video_name}_cnn_features.npy"),
                        np.asarray(self.last_cnn_features),
                    )
                if hasattr(self, "last_lgbm_features") and len(self.last_lgbm_features) > 0:
                    np.save(
                        os.path.join(output_folder, f"{video_name}_lightgbm_features.npy"),
                        np.asarray(self.last_lgbm_features),
                    )

                labeled_path = os.path.join(output_folder, f"labeled_{video_name}")
                label_video(
                    video_path,
                    segments,
                    labeled_path,
                    predictions,
                    valid_timestamps=timestamps,
                    target_fps=25.0,
                )

            elan_path = os.path.join(output_folder, f"{video_name}.eaf")
            create_elan_file(
                video_path,
                segments,
                elan_path,
                fps=get_video_fps(None, video_path),
                include_ground_truth=False,
            )

            return {
                "stats": stats,
                "output_path": elan_path,
            }
        except Exception as error:
            return {"error": str(error)}

    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        video_pattern: str = "*.mp4",
        stride: int = 1,
    ) -> Dict[str, Dict[str, object]]:
        """Process every matching video with both models."""
        os.makedirs(output_folder, exist_ok=True)
        results = {}
        for video_path in glob.glob(os.path.join(input_folder, video_pattern)):
            video_name = os.path.basename(video_path)
            results[video_name] = self.process_video(
                video_path,
                output_folder,
                stride=stride,
            )
        return results

    def predict_labels_from_landmarks(
        self,
        landmarks_per_frame: np.ndarray,
        fps: float,
        stride: int = 1,
    ) -> pd.DataFrame:
        """Predict and combine labels from pre-extracted world landmarks."""
        landmarks = np.asarray(landmarks_per_frame)
        if landmarks.ndim != 2 or landmarks.shape[1] != 92:
            raise ValueError("Expected landmarks with shape (n_frames, 92).")
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")

        cnn_results = self.cnn_detector.model._predict_video_from_landmarks(
            landmarks,
            stride=stride,
        )
        cnn_results = cnn_results.copy()
        cnn_results["timestamp"] = cnn_results["frame_index"] / fps

        lightgbm_model = self.lgbm_detector.model
        lightgbm_model.reset_buffer()
        lgbm_rows = []
        for frame_index, frame_landmarks in enumerate(landmarks[::stride]):
            features = lightgbm_model.extract_features_from_landmarks(frame_landmarks)
            if features is None:
                continue

            probabilities = lightgbm_model.predict(features)[0]
            class_names = list(lightgbm_model.gesture_labels)
            probability_by_label = dict(zip(class_names, probabilities))
            gesture_probability = float(probability_by_label.get("Gesture", 0.0))
            no_gesture_probability = float(
                probability_by_label.get("NoGesture", 0.0)
            )
            prediction = (
                "Gesture"
                if gesture_probability >= lightgbm_model.confidence_threshold
                else "NoGesture"
            )

            source_frame_index = frame_index * stride
            lgbm_rows.append({
                "frame_index": source_frame_index,
                "prediction": prediction,
                "confidence": max(gesture_probability, no_gesture_probability),
                "motion_confidence": gesture_probability,
                "gesture_confidence": gesture_probability,
                "no_gesture_confidence": no_gesture_probability,
                "move_confidence": 0.0,
                "timestamp": source_frame_index / fps,
            })

        lgbm_results = pd.DataFrame(lgbm_rows)
        if lgbm_results.empty:
            lgbm_results = pd.DataFrame(columns=[
                "frame_index", "prediction", "confidence",
                "motion_confidence", "gesture_confidence",
                "no_gesture_confidence", "move_confidence", "timestamp",
            ])

        return self.combine_frame_predictions(
            cnn_results,
            lgbm_results,
            cnn_weight=self.cnn_weight,
            lgbm_weight=self.lgbm_weight,
        )

    def reset(self) -> None:
        """Reset state held by both underlying model detectors."""
        if hasattr(self.cnn_detector.model, "reset_buffer"):
            self.cnn_detector.model.reset_buffer()
        if hasattr(self.lgbm_detector.model, "reset_buffer"):
            self.lgbm_detector.model.reset_buffer()

    def retrack_gestures(
        self,
        input_folder: str,
        output_folder: str,
    ) -> Dict[str, str]:
        """Retrack gesture segments using the existing detector utility."""
        return self.cnn_detector.retrack_gestures(input_folder, output_folder)

    def analyze_dtw_kinematics(
        self,
        landmarks_folder: str,
        output_folder: str,
        fps: float = 25.0,
    ) -> Dict[str, str]:
        """Run DTW and kinematic analysis using the existing utility."""
        return self.cnn_detector.analyze_dtw_kinematics(
            landmarks_folder,
            output_folder,
            fps,
        )

    def prepare_gesture_dashboard(
        self,
        data_folder: str,
        assets_folder: Optional[str] = None,
    ) -> None:
        """Prepare the existing gesture dashboard for combined outputs."""
        self.cnn_detector.prepare_gesture_dashboard(data_folder, assets_folder)

    def combine_frame_predictions(
        self,
        cnn_results: pd.DataFrame,
        lgbm_results: pd.DataFrame,
        cnn_weight: float = 0.5,
        lgbm_weight: float = 0.5,
    ) -> pd.DataFrame:
        """Merge CNN and LightGBM predictions by source video frame.

        The two models use different temporal windows, so rows are aligned by
        ``frame_index``. Rows available from only one model are retained and use
        that model's probabilities.

        The canonical output columns match ``state.Row``. Model-prefixed columns
        are retained so callers can compare the individual predictions.
        """
        if cnn_weight < 0 or lgbm_weight < 0 or cnn_weight + lgbm_weight == 0:
            raise ValueError("Model weights must be non-negative and not both zero.")

        def prepare_results(results: pd.DataFrame, prefix: str) -> pd.DataFrame:
            prepared = results.copy()

            if "frame_index" not in prepared.columns:
                raise ValueError(f"{prefix} results must contain 'frame_index'")

            renamed = {
                column: f"{prefix}_{column}"
                for column in prepared.columns
                if column not in {"frame_index", "timestamp"}
            }
            return prepared.rename(columns=renamed)

        cnn = prepare_results(cnn_results, "cnn")
        lgbm = prepare_results(lgbm_results, "lgbm")
        merged = pd.merge(cnn, lgbm, on="frame_index", how="outer", suffixes=("", "_lgbm"))

        if "timestamp" not in merged.columns:
            merged["timestamp"] = np.nan
        if "timestamp_lgbm" in merged.columns:
            merged["timestamp"] = merged["timestamp"].fillna(merged["timestamp_lgbm"])

        def get_probability(prefix: str, *names: str) -> pd.Series:
            for name in names:
                column = f"{prefix}_{name}"
                if column in merged.columns:
                    return pd.to_numeric(merged[column], errors="coerce")
            return pd.Series(np.nan, index=merged.index, dtype=float)

        cnn_no_gesture = get_probability(
            "cnn", "no_gesture_confidence", "NoGesture_confidence"
        )
        cnn_gesture = get_probability("cnn", "gesture_confidence", "Gesture_confidence")
        cnn_move = get_probability("cnn", "move_confidence", "Move_confidence")
        lgbm_no_gesture = get_probability(
            "lgbm", "no_gesture_confidence", "NoGesture_confidence", "nogesture_prob"
        )
        lgbm_gesture = get_probability(
            "lgbm", "gesture_confidence", "Gesture_confidence", "gesture_prob"
        )

        cnn_available = cnn_gesture.notna() | cnn_no_gesture.notna() | cnn_move.notna()
        lgbm_available = lgbm_gesture.notna() | lgbm_no_gesture.notna()
        cnn_no_gesture = cnn_no_gesture.fillna(0.0)
        cnn_gesture = cnn_gesture.fillna(0.0)
        cnn_move = cnn_move.fillna(0.0)
        lgbm_no_gesture = lgbm_no_gesture.fillna(0.0)
        lgbm_gesture = lgbm_gesture.fillna(0.0)

        combined_no_gesture = cnn_weight * cnn_no_gesture + lgbm_weight * lgbm_no_gesture
        combined_gesture = cnn_weight * cnn_gesture + lgbm_weight * lgbm_gesture
        combined_move = cnn_weight * cnn_move

        active_weight = cnn_weight * cnn_available.astype(float) + lgbm_weight * lgbm_available.astype(float)
        active_weight = active_weight.replace(0.0, np.nan)
        probability_total = combined_no_gesture + combined_gesture + combined_move
        probability_total = probability_total.replace(0.0, np.nan)

        merged["no_gesture_confidence"] = combined_no_gesture / active_weight
        merged["gesture_confidence"] = combined_gesture / active_weight
        merged["move_confidence"] = combined_move / active_weight
        merged["combined_confidence"] = probability_total / active_weight

        probabilities = merged[[
            "no_gesture_confidence",
            "gesture_confidence",
            "move_confidence",
        ]].fillna(0.0)
        labels = np.array(["NoGesture", "Gesture", "Move"])
        # Current fusion policy: select the highest combined probability.
        # TODO: evaluate a calibrated combined threshold or model-agreement rule.
        merged["prediction"] = labels[probabilities.to_numpy().argmax(axis=1)]
        merged["confidence"] = probabilities.max(axis=1)
        merged["motion_confidence"] = (
            merged["gesture_confidence"].fillna(0.0)
            + merged["move_confidence"].fillna(0.0)
        )

        if "timestamp" not in merged.columns:
            merged["timestamp"] = np.nan

        return merged.sort_values("frame_index").reset_index(drop=True)