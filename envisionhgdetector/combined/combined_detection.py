import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

from envisionhgdetector import GestureDetector
from envisionhgdetector.state import Labels, ModelNames, PredictionColumns, StatsKeys, Thresholds
from envisionhgdetector.utils import create_elan_file, create_segments, get_video_fps, label_video

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
        self.cnn_detector = GestureDetector(model_type=ModelNames.CNN_B, config_path=cnn_config_path, weights_path=cnn_weights_path, thresholds=cnn_thresholds)
        self.lgbm_detector = GestureDetector(model_type=ModelNames.LIGHTGBM, config_path=lightgbm_config_path, weights_path=lightgbm_weights_path, thresholds=lightgbm_thresholds)
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
        segments = create_segments(
            segment_input,
            # Use the more conservative thresholds from both models for segmenting
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
            StatsKeys.MODEL_TYPE: "combined",
            "cnn_stats": cnn_stats,
            "lightgbm_stats": lgbm_stats,
            StatsKeys.AVERAGE_MOTION: float(combined_results["motion_confidence"].mean()),
            StatsKeys.AVERAGE_GESTURE: float(combined_results["gesture_confidence"].mean()),
            StatsKeys.AVERAGE_MOVE: float(combined_results["move_confidence"].mean()),
            StatsKeys.CNN_WEIGHT: self.cnn_weight,
            StatsKeys.LIGHTGBM_WEIGHT: self.lgbm_weight,
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
        """Save fused predictions/segments and per-model segments for later rendering."""
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        os.makedirs(output_folder, exist_ok=True)
        video_name, video_extension = os.path.splitext(os.path.basename(video_path))

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

                for model_name, detector in (
                    (ModelNames.CNN_B, self.cnn_detector),
                    (ModelNames.LIGHTGBM, self.lgbm_detector),
                ):
                    prediction_column = f"{model_name}_{PredictionColumns.PREDICTION}"
                    model_predictions = predictions[[
                        PredictionColumns.FRAME_INDEX,
                        PredictionColumns.TIMESTAMP,
                        prediction_column,
                    ]].rename(columns={prediction_column: PredictionColumns.PREDICTION})
                    model_segments = create_segments(
                        model_predictions,
                        min_gap_s=detector.config.thresholds.min_gap_s,
                        min_length_s=detector.config.thresholds.min_length_s,
                        segments_policy="separate",
                    )
                    model_segments_path = os.path.join(
                        output_folder, f"{video_name}_{model_name}_segments.csv"
                    )
                    model_segments.to_csv(model_segments_path, index=False)

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


            elan_path = os.path.join(output_folder, f"{video_name}.eaf")
            create_elan_file(
                video_path=video_path,
                output_path=elan_path,
                segments_df=segments,
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
        cnn_results = self.cnn_detector.model.predict_video_from_landmarks(
            landmarks,
            stride=stride,
            fps=fps,
        )
        cnn_results = cnn_results.copy()
        cnn_results["timestamp"] = cnn_results["frame_index"] / fps

        lgbm_results = self.lgbm_detector.model.predict_video_from_landmarks(
            landmarks,
            fps=fps,
            stride=stride,
        )

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

        Both models must provide predictions for the same unique frame indices.
        Rows are aligned by ``frame_index`` regardless of their input order.

        The canonical output columns match ``state.Row``. Model-prefixed columns
        are retained so callers can compare the individual predictions.
        """
        if cnn_weight < 0 or lgbm_weight < 0 or cnn_weight + lgbm_weight == 0:
            raise ValueError("Model weights must be non-negative and not both zero.")

        def prepare_results(results: pd.DataFrame, prefix: str) -> pd.DataFrame:
            prepared = results.copy()

            for column in (PredictionColumns.FRAME_INDEX, PredictionColumns.PREDICTION):
                if column not in prepared.columns:
                    raise ValueError(f"{prefix} results must contain '{column}' column.")
                if prepared[column].isna().any():
                    raise ValueError(f"{prefix} results contain missing '{column}' values.")
            if prepared[PredictionColumns.FRAME_INDEX].duplicated().any():
                raise ValueError(f"{prefix} results contain duplicate frame indices.")

            renamed = {
                column: f"{prefix}_{column}"
                for column in prepared.columns
                if column not in {PredictionColumns.FRAME_INDEX, PredictionColumns.TIMESTAMP}
            }
            return prepared.rename(columns=renamed)

        cnn = prepare_results(cnn_results, ModelNames.CNN_B)
        lgbm = prepare_results(lgbm_results, ModelNames.LIGHTGBM)
        cnn_frames = pd.Index(cnn[PredictionColumns.FRAME_INDEX])
        lgbm_frames = pd.Index(lgbm[PredictionColumns.FRAME_INDEX])
        missing_cnn = lgbm_frames.difference(cnn_frames)
        missing_lgbm = cnn_frames.difference(lgbm_frames)
        if len(missing_cnn) or len(missing_lgbm):
            raise ValueError(
                "Model frame indices must match: "
                f"CNN is missing {len(missing_cnn)} frames; "
                f"LightGBM is missing {len(missing_lgbm)} frames."
            )
        merged = pd.merge(cnn, lgbm, on=PredictionColumns.FRAME_INDEX, how="inner", validate="one_to_one", suffixes=("", f"_{ModelNames.LIGHTGBM}"))

        if PredictionColumns.TIMESTAMP not in merged.columns:
            merged[PredictionColumns.TIMESTAMP] = np.nan
        if f"{PredictionColumns.TIMESTAMP}_{ModelNames.LIGHTGBM}" in merged.columns:
            merged[PredictionColumns.TIMESTAMP] = merged[PredictionColumns.TIMESTAMP].fillna(merged[f"{PredictionColumns.TIMESTAMP}_{ModelNames.LIGHTGBM}"])

        def get_probability(prefix: str, *names: str) -> pd.Series:
            for name in names:
                column = f"{prefix}_{name}"
                if column in merged.columns:
                    return pd.to_numeric(merged[column], errors="coerce")
            return pd.Series(np.nan, index=merged.index, dtype=float)

        cnn_no_gesture = get_probability(
            ModelNames.CNN_B, PredictionColumns.NO_GESTURE_CONFIDENCE
        )
        cnn_gesture = get_probability(ModelNames.CNN_B, PredictionColumns.GESTURE_CONFIDENCE)
        cnn_move = get_probability(ModelNames.CNN_B, PredictionColumns.MOVE_CONFIDENCE)
        lgbm_no_gesture = get_probability(
            ModelNames.LIGHTGBM, PredictionColumns.NO_GESTURE_CONFIDENCE
        )
        lgbm_gesture = get_probability(
            ModelNames.LIGHTGBM, PredictionColumns.GESTURE_CONFIDENCE
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

        merged[PredictionColumns.NO_GESTURE_CONFIDENCE] = combined_no_gesture / active_weight
        merged[PredictionColumns.GESTURE_CONFIDENCE] = combined_gesture / active_weight
        merged[PredictionColumns.MOVE_CONFIDENCE] = combined_move / active_weight
        merged['COMBINED_CONFIDENCE'] = probability_total / active_weight

        probabilities = merged[[
            PredictionColumns.NO_GESTURE_CONFIDENCE,
            PredictionColumns.GESTURE_CONFIDENCE,
            PredictionColumns.MOVE_CONFIDENCE,
        ]].fillna(0.0)
        NoGesture_label = Labels.NOGESTURE
        Gesture_label = Labels.GESTURE
        Move_label = Labels.MOVE
        labels = np.array([NoGesture_label, Gesture_label, Move_label]) # array so we can index into it with argmax
        print(f"Labels for combined predictions: {labels}")
        # Current fusion policy: select the highest combined probability.
        # TODO: evaluate a calibrated combined threshold or model-agreement rule.
        print(probabilities.to_numpy().argmax(axis=1))
        merged[PredictionColumns.PREDICTION] = labels[probabilities.to_numpy().argmax(axis=1)]
        print('done')
        merged[PredictionColumns.CONFIDENCE] = probabilities.max(axis=1)
        merged[PredictionColumns.MOTION_CONFIDENCE] = (
            merged[PredictionColumns.GESTURE_CONFIDENCE].fillna(0.0)
            + merged[PredictionColumns.MOVE_CONFIDENCE].fillna(0.0)
        )

        if PredictionColumns.TIMESTAMP not in merged.columns:
            merged[PredictionColumns.TIMESTAMP] = np.nan

        return merged.sort_values(PredictionColumns.FRAME_INDEX).reset_index(drop=True)