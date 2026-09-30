"""Find video inputs for gesture analysis."""
from typing import List
from pathlib import Path

def find_all_videos(folder: str | Path, pattern: str = ".mp4") -> List[str]:
    """Recursively return video paths below ``folder``."""
    return [
        str(path)
        for path in Path(folder).rglob("*")
        if path.is_file() and path.name.endswith(pattern)
    ]