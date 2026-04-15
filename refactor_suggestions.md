# Refactor Suggestions for Combined Gesture Model

## Goal
Reduce duplication between `model_combined.py`, `src/models/cnn.py`, and `src/models/lightgbm.py` while keeping the combined-model performance benefit of a single MediaPipe pass.

## Current Division of Responsibility

### Keep in `model_combined.py`
- Shared video traversal
- Single MediaPipe landmark extraction
- Shared frame buffer for the combined pipeline
- Coordination of CNN and LightGBM predictions
- Packaging combined results
- Threshold routing and final output formatting

### Reuse from `src/models/cnn.py`
- CNN model construction
- CNN weight loading
- CNN inference on a prepared 25-frame sequence
- CNN class/confidence interpretation

### Reuse from `src/models/lightgbm.py`
- 100-feature sequence extraction
- LightGBM model loading
- LightGBM inference on a 5-frame sequence
- LightGBM class/confidence interpretation

## What Can Be Replaced

### In `model_combined.py`
1. Replace the internal CNN loading and inference logic with a call into `CNNGestureModel`.
2. Replace the internal LightGBM feature and prediction logic with calls into `LightGBMGestureModel` methods.
3. Remove duplicate LightGBM feature extraction once a shared helper exists.
4. Remove duplicate CNN architecture code if combined can use the same factory as the standalone CNN.

## What Should Stay Custom in Combined Mode

### Combined-specific logic
- Single pass over video frames
- One MediaPipe Holistic instance
- One shared landmark buffer for the combined pipeline
- Slicing the last 5 frames for LightGBM from the shared sequence
- Building the 25-frame CNN input from the same buffer

### Do not delegate directly if it duplicates MediaPipe work
- `predict_frame` from LightGBM should not be used directly in combined mode because it re-extracts landmarks from a frame.
- Any CNN helper that assumes it owns frame traversal should be avoided in combined mode.

## Recommended Refactor Order

### Phase 1: Shared feature extraction
- Move `extract_lgbm_features` into a shared utility module.
- Import that helper from both LightGBM and combined code.
- This is the safest cleanup because the function is already functionally identical.

### Phase 2: Shared CNN builder
- Make `make_model` the canonical CNN builder.
- Have combined import and reuse that same factory instead of rebuilding CNN layers.
- Decide whether preprocessing belongs in the canonical CNN path or should be configurable.

### Phase 3: Combined orchestration only
- Keep `CombinedGestureModel` focused on traversal and orchestration.
- Let the standalone classes handle model-specific behavior.
- Remove duplicate threshold parsing and duplicate prediction post-processing if possible.

## Consistency Decision Needed
The main behavioral difference to settle before refactoring is CNN preprocessing:
- Standalone CNN currently applies preprocessing.
- Combined CNN path currently skips it.

That should be either:
- made identical in both paths, or
- made an explicit config option.

## Practical Implementation Target
A good long-term split is:
- `src/models/cnn.py` as the canonical CNN implementation
- `src/models/lightgbm.py` as the canonical LightGBM implementation
- `model_combined.py` as a thin orchestration layer only

That removes duplication and lowers the risk of model drift.
