"""Shared MediaPipe setup and frame processing."""

import cv2
import mediapipe as mp

holistic = mp.solutions.holistic
pose = mp.solutions.pose
drawing_utils = mp.solutions.drawing_utils


class _MediaPipeProcessor:
    """Own one MediaPipe solution instance for a video or camera session."""

    def __init__(
        self,
        solution,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **options,
    ):
        self._solution = solution
        self.options = dict(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            **options,
        )
        self._instance = None

    def __enter__(self):
        self._instance = self._solution(**self.options)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if self._instance is not None:
                self._instance.close()
        finally:
            self._instance = None

    def process_frame(self, bgr_frame, readonly: bool = False):
        """Convert a BGR frame to RGB and return the raw MediaPipe result."""
        if self._instance is None:
            raise RuntimeError("MediaPipe processor must be opened before processing frames")
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if readonly:
            rgb_frame.flags.writeable = False
        return self._instance.process(rgb_frame)


class HolisticProcessor(_MediaPipeProcessor):
    """Own one MediaPipe Holistic instance for a video or camera session."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **options,
    ):
        super().__init__(
            holistic.Holistic,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            **options,
        )


class PoseProcessor(_MediaPipeProcessor):
    """Own one MediaPipe Pose instance for a video."""

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **options,
    ):
        super().__init__(
            pose.Pose,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            **options,
        )
