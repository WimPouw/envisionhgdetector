"""Shared MediaPipe Holistic setup and frame processing."""

import cv2
import mediapipe as mp

holistic = mp.solutions.holistic
drawing_utils = mp.solutions.drawing_utils


class HolisticProcessor:
    """Own one MediaPipe Holistic instance for a video or camera session."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **options,
    ):
        self.options = dict(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            **options,
        )
        self._holistic = None

    def __enter__(self):
        self._holistic = holistic.Holistic(**self.options)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._holistic is not None:
                self._holistic.close()
        finally:
            self._holistic = None

    def process_frame(self, bgr_frame, readonly: bool = False):
        """Convert a BGR frame to RGB and return the raw MediaPipe result."""
        if self._holistic is None:
            raise RuntimeError("HolisticProcessor must be opened before processing frames")
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if readonly:
            rgb_frame.flags.writeable = False
        return self._holistic.process(rgb_frame)
