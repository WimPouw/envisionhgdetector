
import numpy as np
import pandas as pd

from envisionhgdetector.state import Labels, PredictionColumns

# TODO check if cnn and lightgbm expand consistently- have 1 utils for both

def expand_predictions_to_frames(
        predictions: pd.DataFrame,
        total_frames: int,
        fps: float
    ) -> pd.DataFrame:
        """Return one dataframe row for every source video frame.
        filled unavailable data with NoGesture and prediction_available=False
        """
        frame_df = pd.DataFrame({
            PredictionColumns.FRAME_INDEX: np.arange(total_frames, dtype=np.int64),
        })
        frame_df[PredictionColumns.TIMESTAMP] = frame_df[PredictionColumns.FRAME_INDEX] / fps if fps > 0 else np.nan

        if predictions.empty:
            frame_df[PredictionColumns.PREDICTION] = Labels.NOGESTURE
            frame_df[PredictionColumns.PREDICTION_AVAILABLE] = False
            return frame_df

        dense_df = frame_df.merge(
            predictions,
            on=[PredictionColumns.FRAME_INDEX, PredictionColumns.TIMESTAMP],
            how='left',
        )
        dense_df[PredictionColumns.PREDICTION_AVAILABLE] = dense_df[PredictionColumns.PREDICTION].notna()
        dense_df[PredictionColumns.PREDICTION] = dense_df[PredictionColumns.PREDICTION].fillna(Labels.NOGESTURE)
        return dense_df
    
