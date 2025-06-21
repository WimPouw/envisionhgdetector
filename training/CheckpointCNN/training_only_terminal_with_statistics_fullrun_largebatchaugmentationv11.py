# Standard library imports
import os
import sys
import time
import glob
import importlib.util
from dataclasses import dataclass
from enum import auto, Enum
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple
# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, regularizers, losses
from tensorflow.keras.models import Model
import argparse
from collections import defaultdict  # Added missing import

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Run in test mode')
args = parser.parse_args()

# Change working directory to script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# check gpu availability true false
print(tf.config.list_physical_devices('GPU'))

@dataclass
class Config:
    gesture_labels: Tuple[str, ...] = ("Gesture","Move")#, "Move") #your gesture labels here (dont forget to add a comma)
    undefined_gesture_label: str = "Undefined"
    stationary_label: str = "NoGesture"
    npz_filename: str = "./training/bodylandmarks3.npz" # this is where the training data is stored (mediapipe holistic body from the training videos)
    seq_length: int = 25 # this sets the window size for the classifier, so 17 frames input (so the model needs 15 frames to make a prediction)
    num_original_features: int = 29 # number of features that we originally take in (shown below, be careful this number should match you Feature set)
    weights_filename: str = f"./training/snelliusSAGAZHUBOTEDM3DMULTISIMOECOLANGAUGMENTEDv11.h5" # this contains the CNN model weights already trained with snellius, we will just go on here, but with augmentation

Config = Config()

# only for gpu usage
get_default_weights_path = f"./training/default_weightsv1.h5"

def make_model(weights_path: Optional[str] = get_default_weights_path) -> Model:
    seq_input = layers.Input(
        shape=(Config.seq_length, Config.num_original_features),
        dtype=tf.float32, name="input"
    )
    x = seq_input
    
    # Replace old preprocessing with enhanced version for augmentation
    x = EnhancedPreprocessing()(x)
    
    # block 1
    x = layers.Conv1D(48, 3, strides=1, padding="same", activation="relu")(x)  # Light L2
    x = layers.BatchNormalization()(x)  # Add BN but keep MaxPooling same
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # block 2
    x = layers.Conv1D(96, 3, strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    
    # block 3
    x = layers.Conv1D(192, 3, strides=1, padding="same", activation="relu", 
                     kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Flatten()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x) 

    has_motion = layers.Dense(1, activation="sigmoid", name="has_motion")(x)
    gesture_probs = layers.Dense(len(Config.gesture_labels),
                               activation="softmax", name="gesture_probs")(x)
    output = layers.Concatenate()([has_motion, gesture_probs])

    model = Model(seq_input, output)

    if weights_path is not None:
        if not os.path.isfile(weights_path):
            download_weights_to(weights_path)
        model.load_weights(weights_path)

    return model

class BatchMetricsCallback(tf.keras.callbacks.Callback):
    """Callback to print metrics periodically during training"""
    def __init__(self):
        super(BatchMetricsCallback, self).__init__()
        
    def on_batch_end(self, batch, logs=None):
        if batch % 5000 == 0:  # Print every 5000 batches
            logs = logs or {}
            print(f"\nBatch {batch} - "
                  f"Loss: {logs.get('loss', 0):.4f}, "
                  f"Acc: {logs.get('custom_accuracy', 0):.4f}")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch + 1} complete - "
              f"Loss: {logs.get('loss', 0):.4f}, "
              f"Acc: {logs.get('custom_accuracy', 0):.4f}, "
              f"Val Loss: {logs.get('val_loss', 0):.4f}, "
              f"Val Acc: {logs.get('val_custom_accuracy', 0):.4f}")

def load_landmarks(npz_path: str) -> Dict[str, List[List[float]]]:
    loaded = np.load(npz_path)
    return {label: loaded[label].tolist() for label in loaded.files}

class EnhancedPreprocessing(layers.Layer):
    def __init__(
        self,
        time_warp_range: Tuple[float, float] = (0.9, 1.1),    # Less aggressive time warping
        rotation_range: Tuple[float, float] = (-0.05, 0.05),  # Smaller rotation range
        jitter_sigma: float = 0.005,                          # Smaller jitter for normalized distances
        drop_prob: float = 0.03,                              # Lower drop probability
        noise_stddev: float = 0.01,                           # Lower noise for normalized features
        scale_range: Tuple[float, float] = (0.98, 1.02),      # Smaller scale range
        mask_max_size: int = 2                                # Shorter mask length
    ) -> None:
        super(EnhancedPreprocessing, self).__init__(name="enhanced_preprocessing")
        self.time_warp_range = time_warp_range
        self.rotation_range = rotation_range
        self.jitter_sigma = jitter_sigma
        self.drop_prob = drop_prob
        self.noise_stddev = noise_stddev
        self.scale_range = scale_range
        self.mask_max_size = mask_max_size

    def time_warp(self, features: tf.Tensor) -> tf.Tensor:
        """Apply random temporal warping."""
        warp = tf.random.uniform([], self.time_warp_range[0], self.time_warp_range[1])
        seq_len = tf.shape(features)[1]
        warped = tf.image.resize(
            features[:, :, :, tf.newaxis], 
            [seq_len, tf.shape(features)[2]]
        )[:, :, :, 0]
        return warped

    def add_position_jitter(self, features: tf.Tensor) -> tf.Tensor:
        """Add random jitter to positions."""
        jitter = tf.random.normal(tf.shape(features), mean=0.0, stddev=self.jitter_sigma)
        return features + jitter

    def random_frame_drop(self, features: tf.Tensor) -> tf.Tensor:
        """Randomly drop frames to simulate tracking failures."""
        mask = tf.random.uniform(tf.shape(features)[:2]) > self.drop_prob
        mask = tf.cast(mask, features.dtype)
        return features * mask[:, :, tf.newaxis]

    def time_masking(self, features: tf.Tensor) -> tf.Tensor:
        """Apply random time masking."""
        batch_size = tf.shape(features)[0]
        seq_len = tf.shape(features)[1]
        feature_dim = tf.shape(features)[2]
        
        mask_size = tf.random.uniform([], 1, self.mask_max_size, dtype=tf.int32)
        starts = tf.random.uniform([batch_size], 0, seq_len - mask_size, dtype=tf.int32)
        
        mask = tf.ones([batch_size, seq_len, feature_dim])
        
        for i in range(batch_size):
            start = starts[i]
            indices = tf.range(start, start + mask_size)
            mask_updates = tf.zeros([mask_size, feature_dim])
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.stack([tf.repeat(i, mask_size), indices], axis=1),
                mask_updates
            )
        
        return features * mask

    def compute_derivatives(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute first and second derivatives."""
        # First derivative
        t_deriv = features[:, 1:] - features[:, :-1]
        t_deriv = tf.pad(t_deriv, [[0, 0], [1, 0], [0, 0]])

        # Second derivative
        t_deriv_2 = t_deriv[:, 1:] - t_deriv[:, :-1]
        t_deriv_2 = tf.pad(t_deriv_2, [[0, 0], [1, 0], [0, 0]])
        
        return t_deriv, t_deriv_2

    def compute_statistics(
        self, 
        features: tf.Tensor, 
        t_deriv: tf.Tensor, 
        t_deriv_2: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute statistical features."""
        features_std = tf.math.reduce_std(features, axis=-2, keepdims=True)
        t_deriv_std = tf.math.reduce_std(t_deriv, axis=-2, keepdims=True)
        t_deriv_std_2 = tf.math.reduce_std(t_deriv_2, axis=-2, keepdims=True)
        
        return features_std, t_deriv_std, t_deriv_std_2

    def normalize_features(self, features: tf.Tensor) -> tf.Tensor:
        """Apply feature normalization."""
        mean = tf.reduce_mean(features, axis=-1, keepdims=True)
        std = tf.math.reduce_std(features, axis=-1, keepdims=True) + 1e-8
        return (features - mean) / std

    def call(
        self, 
        x: tf.Tensor,
        training: Optional[bool] = None,
        mask: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        features = x
        
        # Center the features
        features = features - tf.reduce_mean(features, axis=-2, keepdims=True)
        
        # Compute derivatives and statistics
        t_deriv, t_deriv_2 = self.compute_derivatives(features)
        features_std, t_deriv_std, t_deriv_std_2 = self.compute_statistics(
            features, t_deriv, t_deriv_2
        )
        
        # Concatenate all features
        features = tf.concat([
            features,
            t_deriv,
            t_deriv_2,
            tf.broadcast_to(features_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std, tf.shape(features)),
            tf.broadcast_to(t_deriv_std_2, tf.shape(features))
        ], axis=-1)
        
        if training:
            # Apply augmentations
            features = self.time_warp(features)
            features = self.add_position_jitter(features)
            features = self.random_frame_drop(features)
            
            # Add noise
            features = features + tf.random.normal(
                tf.shape(features), 
                mean=0.0, 
                stddev=self.noise_stddev
            )
            
            # Random scaling
            scale = tf.random.uniform(
                [], 
                self.scale_range[0], 
                self.scale_range[1], 
                dtype=tf.float32
            )
            features = features * scale
            
            # Apply time masking
            features = self.time_masking(features)
        
        # Final normalization
        features = self.normalize_features(features)
        
        return features
    
def setup_accelerators_and_get_strategy() -> tf.distribute.Strategy:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        # Strategy for GPU or multi-GPU machines.
        strategy = tf.distribute.MirroredStrategy()
        print(f"Number of accelerators: {strategy.num_replicas_in_sync}")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        print("Using single CPU.")
    return strategy

def make_y(label_idx: int) -> List[int]:
    has_motion = 1 if label_idx > 0 else 0
    y = [has_motion] + [0] * len(Config.gesture_labels)
    if has_motion == 1:
        y[label_idx] = 1
    # Note:
    # Format of y is [has_motion, <one-hot-gesture-class-vector>], e.g.,
    # [1, 0, 1, ..., 0] for the 2nd gesture,
    # [0, ..., 0] for stationary case.
    return y


def make_ds_train(
        landmark_dict: Dict[str, Sequence[Sequence[float]]],
        seq_length: int,
        num_features: int,
        seed: int
) -> tf.data.Dataset:
    # Note: stationary label must come first in this design, see make_y.
    labels = (Config.stationary_label,) + Config.gesture_labels
    rng = np.random.default_rng(seed=seed)

    def gen() -> Tuple[List[List[float]], int]:
        while True:
            label_idx = int(rng.integers(len(labels), size=1))
            landmarks = landmark_dict[labels[label_idx]]
            seq_idx = int(rng.integers(len(landmarks) - seq_length, size=1))
            features = landmarks[seq_idx: seq_idx + seq_length]
            yield features, make_y(label_idx)

    # Create dataset
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_length, num_features), dtype=tf.float32),
            tf.TensorSpec(shape=(len(labels),), dtype=tf.int32)
        )
    )

    # Add data sharding options
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    return ds

def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    has_motion_true = y_true[:, :1]
    has_motion_pred = y_pred[:, :1]
    has_motion_loss = losses.BinaryCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )(has_motion_true, has_motion_pred)

    # The gesture loss is designed in the way that, if has_motion is 0,
    # the box values do not matter.
    mask = y_true[:, 0] == 1
    weight = tf.where(mask, 1.0, 0.0)
    gesture_true = y_true[:, 1:]
    gesture_pred = y_pred[:, 1:]
    gesture_loss = losses.CategoricalCrossentropy(
        label_smoothing=0.05,
        reduction=tf.keras.losses.Reduction.NONE
    )(gesture_true, gesture_pred, sample_weight=weight)

    return (has_motion_loss + gesture_loss) * 0.5


class CustomAccuracy(tf.keras.metrics.Metric):

    def __init__(
            self,
            motion_threshold: float = 0.5,
            gesture_threshold: float = 0.9,
            name: str = "custom_accuracy"
    ) -> None:
        super(CustomAccuracy, self).__init__(name=name)
        self.motion_threshold = motion_threshold
        self.gesture_threshold = gesture_threshold
        self.acc = tf.keras.metrics.CategoricalAccuracy()

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None
    ) -> None:
        # IMPORTANT - the sample_weight parameter is needed to solve:
        # TypeError: tf__update_state() got an unexpected keyword argument 'sample_weight'
        y_pred = tf.where(y_pred[:, :1] >= self.motion_threshold, y_pred, 0.0)
        y_pred = tf.where(y_pred >= self.gesture_threshold, 1.0, 0.0)
        self.acc.update_state(y_true[:, 1:], y_pred[:, 1:])

    def result(self) -> tf.Tensor:
        return self.acc.result()

    def reset_state(self) -> None:
        self.acc.reset_state()


def compile_model(model: Model) -> None:
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(amsgrad=True),
        metrics=[CustomAccuracy()],
    )

#def get_steps_per_epoch(
#        landmark_dict: Dict[str, Sequence[Sequence[float]]]
#) -> int:
#    # Kind of arbitrary here.
#    mean_data_size = int(np.mean([len(v) for v in landmark_dict.values()]))
#    steps_per_epoch = int(mean_data_size * 0.7)
#    return steps_per_epoch

def get_steps_per_epoch(
        landmark_dict: Dict[str, Sequence[Sequence[float]]]
) -> int:
    # Fixed smaller number of steps that should complete within SLURM time limit
    return 100000  # This is about 1/50th of your current step

class PeriodicSaveCallback(tf.keras.callbacks.Callback):
    """Save metrics and model weights periodically"""
    def __init__(self, save_dir='./training', save_freq_minutes=30):
        super(PeriodicSaveCallback, self).__init__()
        self.save_dir = save_dir
        self.save_freq_seconds = save_freq_minutes * 60
        self.last_save_time = None
        self.history_backup = defaultdict(list)
        
    def on_train_begin(self, logs=None):
        self.last_save_time = time.time()
        
    def save_training_state(self, epoch, logs):
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Update history backup
        if logs is not None:
            for metric, value in logs.items():
                self.history_backup[metric].append(value)
        
        # Save metrics to CSV
        pd.DataFrame(self.history_backup).to_csv(
            f'{self.save_dir}/metrics_{timestamp}.csv',
            index=False
        )
        
        # Plot training curves
        if len(self.history_backup['loss']) > 0:  # Only plot if we have data
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history_backup['loss'], label='Training Loss')
            if 'val_loss' in self.history_backup:
                plt.plot(self.history_backup['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(self.history_backup['custom_accuracy'], label='Training Accuracy')
            if 'val_custom_accuracy' in self.history_backup:
                plt.plot(self.history_backup['val_custom_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{self.save_dir}/training_history_{timestamp}.png')
            plt.close()
        
        # Save model weights
        self.model.save_weights(f'{self.save_dir}/model_weights_{timestamp}.h5')
        
    def on_epoch_end(self, epoch, logs=None):
        current_time = time.time()
        if current_time - self.last_save_time >= self.save_freq_seconds:
            self.save_training_state(epoch, logs)
            self.last_save_time = current_time

def train_and_save_weights(
        landmark_dict: Dict[str, List[List[float]]],
        model: Model,
        weights_path: str,
        seed: int = 42
) -> None:
    # Set a safe batch size for A100
    BATCH_SIZE = 2048  # Safe batch size for A100 with your model
    
    # Training dataset
    ds_train = make_ds_train(
        landmark_dict, Config.seq_length, Config.num_original_features, seed
    )
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Validation dataset from same source with different seed
    ds_val = make_ds_train(
        landmark_dict, Config.seq_length, Config.num_original_features, seed+1
    )
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Calculate steps
    steps_per_epoch = get_steps_per_epoch(landmark_dict)
    validation_steps = steps_per_epoch // 5  # 20% validation

    # Setup callbacks
    callbacks = [
        PeriodicSaveCallback(save_freq_minutes=30),  # Save every 30 minutes
        BatchMetricsCallback(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-04,
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]
    
    # Enable mixed precision for better performance
    #tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    history = model.fit(
        ds_train,
        epochs=500,
        steps_per_epoch=steps_per_epoch,
        validation_data=ds_val,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final save at the end of training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pd.DataFrame(history.history).to_csv(f'./training/final_metrics_{timestamp}.csv', index=False)
    
    return history

def main(test_run: bool = True) -> None:
    # Enable mixed precision for better performance
    #tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    strategy = setup_accelerators_and_get_strategy()
    with strategy.scope():
        model = make_model(weights_path=None)
        compile_model(model)
    landmark_dict = load_landmarks(Config.npz_filename)
    
    try:
        if test_run:
            print("Running test training with 5 epochs...")
            # Create test versions of filenames
            test_weights = Config.weights_filename.replace('.h5', '_test.h5')
            
            # Test training dataset with 512 batch size
            ds_train = make_ds_train(
                landmark_dict, Config.seq_length, Config.num_original_features, 42
            ).batch(16).prefetch(tf.data.AUTOTUNE)
            
            # Test validation dataset
            ds_val = make_ds_train(
                landmark_dict,  # Using same dict for test
                Config.seq_length, Config.num_original_features, 43
            ).batch(16).prefetch(tf.data.AUTOTUNE)
            
            # Test callbacks
            test_callbacks = [
                PeriodicSaveCallback(save_dir='./training', save_freq_minutes=5),  # More frequent saves for testing
                BatchMetricsCallback(),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=test_weights,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                )
            ]
            
            # Modify training parameters for quick test
            history = model.fit(
                ds_train,
                epochs=5,  # Reduced epochs for test
                steps_per_epoch=min(100, get_steps_per_epoch(landmark_dict)),  # Limit steps
                validation_data=ds_val,
                validation_steps=min(20, get_steps_per_epoch(landmark_dict)),  # Limit validation steps
                callbacks=test_callbacks,
                verbose=1
            )
            
            # Save test results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Test Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['custom_accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_custom_accuracy'], label='Validation Accuracy')
            plt.title('Test Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'./training/test_training_history_{timestamp}.png')
            plt.close()
            
            # Save test metrics
            pd.DataFrame(history.history).to_csv(f'./training/test_metrics_{timestamp}.csv', index=False)
            
            print("\nTest run completed. Check test_training_history.png and test_metrics.csv")
            print("If everything looks good, run with test_run=False for full training")
            return
        
        # Full training
        print("Starting full training with periodic saves...")
        history = train_and_save_weights(landmark_dict, model, Config.weights_filename)
        
        # Final save at end of training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['custom_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_custom_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'./training/final_training_history_{timestamp}.png')
        pd.DataFrame(history.history).to_csv(f'./training/final_metrics_{timestamp}.csv', index=False)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        # Save current state on interrupt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if 'history' in locals():
            pd.DataFrame(history.history).to_csv(f'./training/interrupted_metrics_{timestamp}.csv', index=False)
            model.save_weights(f'./training/interrupted_weights_{timestamp}.h5')
        print(f"Saved interrupted state at timestamp: {timestamp}")

if __name__ == "__main__":
    main(test_run=args.test)