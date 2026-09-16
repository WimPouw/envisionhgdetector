import tqdm
import argparse
import numpy as np
from pathlib import Path

from envisionhgdetector import GestureDetector

argparser = argparse.ArgumentParser(description='Process npz files and generate predictions.')
argparser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder containing npz files.')
argparser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder to save predictions and metadata.')
argparser.add_argument('--model', type=str, required=True, help='CNN or LightGBM model to use for predictions. Options: "cnn_b" or "lightgbm".')
args = argparser.parse_args()

model = args.model
if model not in ['cnn_b', 'lightgbm']:
    raise ValueError(f"Invalid model type: {model}. Please choose either 'cnn_b' or 'lightgbm'.")

# using default thresholds
detector = GestureDetector(model_type=model)

input_folder = Path(args.input_folder) 
output_folder = Path(args.output_folder) 

if not input_folder.exists():
    raise FileNotFoundError(f"Input folder {input_folder} does not exist.")

npz_files = list(input_folder.glob("*.npz"))
print(f"Found {len(npz_files)} npz files in {input_folder}.")
output_folder.mkdir(parents=True, exist_ok=True)
print(f"Output folder is set to {output_folder}.")

with tqdm.tqdm(npz_files, desc="Processing npz files") as pbar:
    for file in npz_files:
        file_contents = np.load(file, allow_pickle=False)
        fps = file_contents['fps'].item()  # Extract fps from the npz file
        landmarks = file_contents['world_body_landmarks']

        UPPER_BODY_INDICES = list(range(23)) # we are only interested in the upper body landmarks for gesture detection
        all_features = []

        for frame_landmarks in landmarks: # per frame
            features = []
            for idx in UPPER_BODY_INDICES: # per landmark
                if idx < len(frame_landmarks):
                    lm = frame_landmarks[idx, :]
                    features.extend(lm)
                
            all_features.append(features)

        stripped_features = np.array(all_features, dtype=np.float32)
        results_df = detector.predict_labels_from_landmarks(stripped_features, fps=fps)
        results_df.to_csv(output_folder / f"{file.stem}_predictions.csv", index=False) # save results to csv

        pbar.update(1)  # Update the progress bar after processing each file