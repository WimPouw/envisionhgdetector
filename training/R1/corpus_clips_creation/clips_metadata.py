from pathlib import Path
from datetime import datetime
import json

class ClipsMetadata:
    def __init__(self, directory: Path):
        self.directory = directory

    def return_clip_info_from_path(self, output_path: Path):
        path_stem = output_path.stem
        parts = path_stem.split('_')
        corpus_name = parts[0]
        type = parts[-1]
        label = parts[-2]
        clip_id = parts[-3]
        speaker = '_'.join(parts[1:-3])  # Join all parts between corpus_name and clip_id
        return {
            "corpus_name": corpus_name,
            "type": type,
            "label": label,
            "clip_id": clip_id,
            "speaker": speaker
        }

    def traverse_file(self, json_path: Path):
        labels_info = {}
        types_info = {}
        speakers_info = {}

        unique_types_count = {}
        unique_speakers_count = {}

        with open(json_path, "r") as f:
            video_clips = json.load(f) # list

        for clip in video_clips:
            clip_output_path = Path(clip.get("output_path"))
            corpus_name, type, label, clip_id, speaker = self.return_clip_info_from_path(clip_output_path).values()
            duration = clip.get("end") - clip.get("start")
            unique_types_count[type] = unique_types_count.get(type, 0) + 1
            unique_speakers_count[speaker] = unique_speakers_count.get(speaker, 0) + 1

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

            if speaker not in speakers_info:
                speakers_info[speaker] = {
                    "total_clip_duration": duration,
                    "num_clips": 1
                }
            else:
                speakers_info[speaker]["total_clip_duration"] += duration
                speakers_info[speaker]["num_clips"] += 1


        for label in labels_info:
            labels_info[label]["avg_clip_duration"] = labels_info[label]["total_clip_duration"] / labels_info[label]["num_clips"]
        for type in types_info:
            types_info[type]["avg_clip_duration"] = types_info[type]["total_clip_duration"] / types_info[type]["num_clips"]
        for speaker in speakers_info:
            speakers_info[speaker]["avg_clip_duration"] = speakers_info[speaker]["total_clip_duration"] / speakers_info[speaker]["num_clips"]

        return labels_info, types_info, speakers_info, unique_types_count, unique_speakers_count
    
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
        metadata = {}
        json_files = list(self.directory.glob("*.json"))
        by_corpus = {}

        for video_path in json_files:
            corpus_name = video_path.name.split("_")[0]
            new_labels_info, new_types_info, new_speakers_info, new_unique_types_count, new_unique_speakers_count = self.traverse_file(video_path)

            if corpus_name not in by_corpus:
                by_corpus[corpus_name] = {"by_label": new_labels_info, "by_type": new_types_info, "by_speaker": new_speakers_info, "unique_types_count": new_unique_types_count, "unique_speakers_count": new_unique_speakers_count}
            else:
                self._merge_info(by_corpus[corpus_name]["by_label"], new_labels_info)
                self._merge_info(by_corpus[corpus_name]["by_type"], new_types_info)
                self._merge_info(by_corpus[corpus_name]["by_speaker"], new_speakers_info)

                for type, count in new_unique_types_count.items():
                    by_corpus[corpus_name]["unique_types_count"][type] = by_corpus[corpus_name]["unique_types_count"].get(type, 0) + count
                for speaker, count in new_unique_speakers_count.items():
                    by_corpus[corpus_name]["unique_speakers_count"][speaker] = by_corpus[corpus_name]["unique_speakers_count"].get(speaker, 0) + count

        # Compute per-corpus totals
        for corpus_name, corpus in by_corpus.items():
            corpus["num_clips"] = sum(v["num_clips"] for v in corpus["by_label"].values())
            corpus["total_clip_duration"] = sum(v["total_clip_duration"] for v in corpus["by_label"].values())
            corpus["avg_clip_duration"] = corpus["total_clip_duration"] / corpus["num_clips"] if corpus["num_clips"] else 0
            corpus["total_unique_speakers"] = len(corpus["unique_speakers_count"])
            corpus["total_unique_types"] = len(corpus["unique_types_count"])
            corpus['avg_videos_per_speaker'] = corpus["num_clips"] / corpus["total_unique_speakers"] if corpus["total_unique_speakers"] else 0

        # Global by_label / by_type
        by_label = {}
        by_type = {}
        by_speaker = {}
        for corpus in by_corpus.values():
            self._merge_info(by_label, corpus["by_label"])
            self._merge_info(by_type, corpus["by_type"])
            self._merge_info(by_speaker, corpus["by_speaker"])

        metadata["generated_at"] = datetime.now().isoformat()
        metadata["metadata"] = {
            "num_corpora": len(by_corpus),
            "num_clips": sum(c["num_clips"] for c in by_corpus.values()),
            "total_clip_duration": sum(c["total_clip_duration"] for c in by_corpus.values()),
            "avg_clip_duration": sum(c["total_clip_duration"] for c in by_corpus.values()) / sum(c["num_clips"] for c in by_corpus.values()),
            "unique_speakers_count": sum(len(c["unique_speakers_count"]) for c in by_corpus.values()),
            "unique_types_count": sum(len(c["unique_types_count"]) for c in by_corpus.values())
        }
        metadata["by_corpus"] = by_corpus
        metadata["by_label"] = by_label
        metadata["by_type"] = by_type
        metadata["by_speaker"] = by_speaker

        return metadata