import yaml
import importlib
import logging
import json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from clips_metadata import ClipsMetadata

from utils import check_ffmpeg
"""
tasks
- check corupus folder to see naming conventions of videos and annotations
- update gesres file
- standardize functions and variable names across files
- re-read changes to ensure consistency and correctness
- confirm audio is present in clips
"""


def main():
    if not check_ffmpeg():
        print("Error: ffmpeg not found. Please install ffmpeg and ensure it's in your system's PATH.")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - macOS: brew install ffmpeg")
        print("  - Linux: sudo apt-get install ffmpeg (Ubuntu/Debian)")
        exit(1)
    
    # Load configuration    
    with open('corpus_details.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # create output directories if they don't exist
    defaults = config.get('defaults', {})
    if not defaults:
        raise ValueError("Defaults section is missing in the configuration file.")
    
    output_base_dir = Path(defaults.get('output_directory', 'CorpusClips'))
    output_base_dir.mkdir(exist_ok=True)
    
    current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging_folder = output_base_dir / 'logs'
    logging_folder.mkdir(exist_ok=True)
    logging_file_path = logging_folder / f'corpus_clips_creation_{current_session}.log'
    logging.basicConfig(level=logging.INFO, filename=logging_file_path, format='%(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s',)

    gesture_output_dir = output_base_dir / defaults.get('gesture_output_directory', 'GestureClips')
    no_gesture_output_dir = output_base_dir / defaults.get('no_gesture_output_directory', 'NoGestureClips')
    move_output_dir = output_base_dir / defaults.get('move_output_directory', 'MoveClips')
    clips_info_dir = output_base_dir / defaults.get('clips_info_directory', 'ClipsInfo')
    clips_info_dir.mkdir(exist_ok=True)
    defaults['gesture_output_directory'] = str(gesture_output_dir)
    defaults['no_gesture_output_directory'] = str(no_gesture_output_dir)
    defaults['move_output_directory'] = str(move_output_dir)
    defaults['clips_info_directory'] = str(clips_info_dir)

    gesture_output_dir.mkdir(exist_ok=True)
    no_gesture_output_dir.mkdir(exist_ok=True)
    move_output_dir.mkdir(exist_ok=True)

    corpora = config.get('corpora', [])
    if not corpora:
        raise ValueError("Corpora section is missing or empty in the configuration file.")
    
    corpora_instances = load_corpora(corpora, defaults)
    if len(corpora_instances) == 0:
        raise ValueError("No valid corpora instances were created. Please check the configuration file and ensure that at least one corpus is enabled and correctly specified.")
    else:
        logging.info(f"Successfully created {len(corpora_instances)} corpus instances for processing.")
    
    # Parallel Processing
    num_workers = min(len(corpora_instances), config.get('num_workers', 4))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_corpus = {
            executor.submit(corpus.extract): corpus for corpus in corpora_instances
        }
        for future in as_completed(future_to_corpus):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Faced Exception: {e} while processing corpus: {future_to_corpus[future].name}")
                print(f"Faced Exception: {e} while processing corpus: {future_to_corpus[future].name}")

    # After all corpora are processed, generate metadata
    clips_metadata = ClipsMetadata(clips_info_dir)
    metadata = clips_metadata.save_metadata()
    with open(output_base_dir / "clips_metadata.json", "w+") as f:
        json.dump(metadata, f, indent=4)

    print(f"Processing completed. Metadata saved to {output_base_dir / 'clips_metadata.json'}")

def parse_code_file(code_file: str) -> tuple[str, str]:
    if not code_file or '.' not in code_file:
        raise ValueError(f"Invalid code_file format: {code_file}. Expected format: 'module.path.ClassName'")
    
    parts = code_file.split('.')
    class_name = parts[-1]
    module_path = '.'.join(parts[:-1])
    
    return module_path, class_name

def load_corpora(corpora_specifications: dict, defaults: dict) -> list:
    corpora = []
    for spec in corpora_specifications:
        corpus_name = spec.get("name")
        corpus_dir = Path(spec.get("directory", {}))
        corpus_enabled = spec.get("enabled", True)

        if not corpus_enabled:
            continue

        try:
            module_path, class_name = parse_code_file(spec.get("module", ""))
            corpus_module = importlib.import_module(module_path)
            corpus_class = getattr(corpus_module, class_name)
            corpus = corpus_class(corpus_name, corpus_dir, defaults)
            corpora.append(corpus)
        except (ImportError, AttributeError, TypeError) as e:
            logging.error(f"Error instantiating corpus {corpus_name}: {e}")

    return corpora

if __name__ == "__main__":
    main()