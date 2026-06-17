from pathlib import Path
from datetime import datetime
import json

class ClipsMetadata:
    def __init__(self, directory: Path):
        self.directory = directory

    def traverse_file(self, json_path: Path):
        labels_info = {}
        types_info = {}
        unique_types = {}
        with open(json_path, "r") as f:
            video_clips = json.load(f) # list

        for clip in video_clips:
            label = clip.get("label")
            type = clip.get("type")
            duration = clip.get("end") - clip.get("start")
            unique_types[type] = unique_types.get(type, 0) + 1
            if label not in labels_info:
                labels_info[label] = {
                    "total_clip_duration": duration,
                    "num_clips": 1
                }
            else:
                labels_info[label]["total_clip_duration"] += duration
                labels_info[label]["num_clips"] += 1
            
            if type not in types_info:
                types_info[type] = {
                    "total_clip_duration": duration,
                    "num_clips": 1
                }
            else:
                types_info[type]["total_clip_duration"] += duration
                types_info[type]["num_clips"] += 1

        for label in labels_info:
            labels_info[label]["avg_clip_duration"] = labels_info[label]["total_clip_duration"] / labels_info[label]["num_clips"]
        for type in types_info:
            types_info[type]["avg_clip_duration"] = types_info[type]["total_clip_duration"] / types_info[type]["num_clips"]

        return labels_info, types_info, unique_types
    
    def _merge_info(self, old: dict, new: dict):
        """Merge new info dict into old in place."""
        for key, new_val in new.items():
            if key not in old:
                old[key] = new_val.copy()
            else:
                old[key]["total_clip_duration"] += new_val["total_clip_duration"]
                old[key]["num_clips"] += new_val["num_clips"]
                old[key]["avg_clip_duration"] = old[key]["total_clip_duration"] / old[key]["num_clips"]


    def save_metadata(self):
        """
        generated_at: timestamp,
        metadata: {
            num_corpora: 5,
            num_clips: 100,
        },
        by_label: {
            label_name: {
                avg_clip_duration: 5.0,
                total_clip_duration: 500.0,
                num_clips: 100
            }, ...
        },
        by_type: {
            type_name: {
                avg_clip_duration: 5.0,
                total_clip_duration: 500.0,
                num_clips: 100
            }, ...
        },
        by_corpus: {
            corpus_name: {
                avg_clip_duration: 5.0,
                total_clip_duration: 500.0,
                num_clips: 100,
                by_label: {
                    label_name: {
                        avg_clip_duration: 5.0,
                        total_clip_duration: 500.0,
                        num_clips: 100
                    }, ...
                },
                by_type: {
                    type_name: {
                        avg_clip_duration: 5.0,
                        total_clip_duration: 500.0,
                        num_clips: 100
                    }, ...
                }
            }, ...
        }
        """
        metadata = {}
        json_files = list(self.directory.glob("*.json"))
        by_corpus = {}

        for video_path in json_files:
            corpus_name = video_path.name.split("_")[0]
            new_labels_info, new_types_info, new_unique_types = self.traverse_file(video_path)

            if corpus_name not in by_corpus:
                by_corpus[corpus_name] = {"by_label": new_labels_info, "by_type": new_types_info, "unique_types": new_unique_types}
            else:
                self._merge_info(by_corpus[corpus_name]["by_label"], new_labels_info)
                self._merge_info(by_corpus[corpus_name]["by_type"], new_types_info)
                for type, count in new_unique_types.items():
                    by_corpus[corpus_name]["unique_types"][type] = by_corpus[corpus_name]["unique_types"].get(type, 0) + count

        # Compute per-corpus totals
        for corpus_name, corpus in by_corpus.items():
            corpus["num_clips"] = sum(v["num_clips"] for v in corpus["by_label"].values())
            corpus["total_clip_duration"] = sum(v["total_clip_duration"] for v in corpus["by_label"].values())
            corpus["avg_clip_duration"] = corpus["total_clip_duration"] / corpus["num_clips"] if corpus["num_clips"] else 0

        # Global by_label / by_type
        by_label = {}
        by_type = {}
        for corpus in by_corpus.values():
            self._merge_info(by_label, corpus["by_label"])
            self._merge_info(by_type, corpus["by_type"])

        metadata["generated_at"] = datetime.now().isoformat()
        metadata["metadata"] = {
            "num_corpora": len(by_corpus),
            "num_clips": sum(c["num_clips"] for c in by_corpus.values()),
            "total_clip_duration": sum(c["total_clip_duration"] for c in by_corpus.values()),
            "avg_clip_duration": sum(c["total_clip_duration"] for c in by_corpus.values()) / sum(c["num_clips"] for c in by_corpus.values()),
        }
        metadata["by_corpus"] = by_corpus
        metadata["by_label"] = by_label
        metadata["by_type"] = by_type

        return metadata



# if __name__ == "__main__":
#     directory = Path("C:\\Users\\User\\Desktop\\zipped~\\CorpusClips\\ClipsInfo")
#     clips_metadata = ClipsMetadata(directory)
#     metadata = clips_metadata.save_metadata()
#     with open(directory.parent / "clips_metadata.json", "w+") as f:
#         json.dump(metadata, f, indent=4)