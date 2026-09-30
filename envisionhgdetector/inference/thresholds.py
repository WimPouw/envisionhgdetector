"""Select prediction labels from confidence values."""

import pandas as pd

from ..state import Labels, PredictionColumns

def get_label_from_prediction(
    no_gesture_confidence: float,
    gesture_confidence: float,
    move_confidence: float,
    motion_threshold: float,
    gesture_threshold: float
) -> str:
    """Apply motion and gesture thresholds to confidence values."""
    has_motion = 1 - no_gesture_confidence
    prediction = Labels.NOGESTURE
    
    if has_motion >= motion_threshold:
        gesture_conf = gesture_confidence
        move_conf = move_confidence
        
        valid_gestures = []
        if gesture_conf >= gesture_threshold:
            valid_gestures.append((Labels.GESTURE, gesture_conf))
        if move_conf >= gesture_threshold:
            valid_gestures.append((Labels.MOVE, move_conf))
            
        if valid_gestures:
            prediction = max(valid_gestures, key=lambda x: x[1])[0]
            
    return prediction

def get_prediction_at_threshold(
    row: pd.Series,
    motion_threshold: float,
    gesture_threshold: float
) -> str:
    """Apply thresholds to a prediction row using the scalar helper."""
    return get_label_from_prediction(
        no_gesture_confidence=row[PredictionColumns.NO_GESTURE_CONFIDENCE],
        gesture_confidence=row[PredictionColumns.GESTURE_CONFIDENCE],
        move_confidence=row[PredictionColumns.MOVE_CONFIDENCE],
        motion_threshold=motion_threshold,
        gesture_threshold=gesture_threshold,
    )


