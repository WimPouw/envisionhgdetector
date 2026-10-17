import numpy as np
from pathlib import Path

from envisionhgdetector import GestureDetector

# running on default thresholds - import Thresholds from envisionhgdetector.state if you want to customize thresholds
model = GestureDetector(model_type='cnn_b') # or lightgbm

input_file = Path(NPZ_FILE_PATH)  # Replace with the actual path to your .npz file
output_folder = Path(OUTPUT_FOLDER_PATH)  # Replace with the desired output folder path
output_folder.mkdir(parents=True, exist_ok=True)

if not input_file.exists():
    raise FileNotFoundError(f"Input file {input_file} does not exist.")

file_contents = np.load(input_file, allow_pickle=False)
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

results_df = model.predict_labels_from_landmarks(stripped_features, fps=fps)
results_df.to_csv(output_folder / f"{input_file.stem}_predictions.csv", index=False) # save results to csv

print(f"Number of predictions: {len(results_df)}, Number of input frames: {len(stripped_features)}")