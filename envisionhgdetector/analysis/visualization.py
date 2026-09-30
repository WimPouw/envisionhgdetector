"""Generate coordinates for visualizing gesture similarities."""

import os
from typing import List

import numpy as np
import pandas as pd
import umap.umap_ as umap


def create_gesture_visualization(
    dtw_matrix: np.ndarray,
    gesture_names: List[str],
    output_folder: str,
) -> None:
    """Save a two-dimensional UMAP projection of DTW distances as CSV."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        metric="precomputed",
    )
    projection = reducer.fit_transform(dtw_matrix)
    viz_df = pd.DataFrame({
        "x": projection[:, 0],
        "y": projection[:, 1],
        "gesture": gesture_names,
    })
    viz_df.to_csv(os.path.join(output_folder, "gesture_visualization.csv"), index=False)
