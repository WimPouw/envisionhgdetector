"""Interface template for a gesture model used by the detector layer.

This is a contract for future implementations. Existing models do not inherit
from it yet, and feature extraction remains specific to each model.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ModelTemplate(ABC):
    """Common inference interface for video and landmark inputs."""

    @abstractmethod
    def __init__(self, config: Any) -> None:
        """Store configuration and load the model resources needed for inference."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return raw model scores for a batch of model-ready features.

        The feature shape and score columns depend on the model. Callers should
        use the normalized ``predict_video_from_landmarks`` output when they
        need labels or cross-model confidence columns.
        """

    @abstractmethod
    def predict_video_from_landmarks(
        self, landmarks_per_frame: np.ndarray, fps: float, stride: int = 1
    ) -> pd.DataFrame:
        """Return timestamped predictions from pre-extracted landmarks.

        ``fps`` is the source frame rate and ``stride`` is the sampling step.
        The result should use the columns in ``state.PredictionColumns``:
        frame_index, prediction, confidence, motion_confidence,
        gesture_confidence, no_gesture_confidence, move_confidence, timestamp.
        A model without a Move class should report move_confidence as zero.
        Return an empty DataFrame with these columns when no predictions exist.
        """

    @abstractmethod
    def predict_video(
        self, video_path: str, stride: int = 1
    ) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, np.ndarray, list[float]]:
        """Process a video and return predictions, statistics, segments, features, timestamps.

        Predictions use ``state.PredictionColumns``. Segments use start_time,
        end_time, labelid, label, and duration. ``features`` contains the
        extracted model features; ``timestamps`` identifies their source times
        in seconds. Keep this five-item result even when no frames are usable.
        """



