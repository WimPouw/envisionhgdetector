"""Find video inputs for gesture analysis."""

import os
from typing import List


def find_all_videos(folder: str, pattern: str = ".mp4") -> List[str]:
    """Recursively return video paths below ``folder``."""
    videos = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(pattern):
                videos.append(os.path.join(root, file))
    return videos
