import yaml
from typing import Literal
from pathlib import Path

from .state import CNN_B_Config, LIGHTGBM_Config, Thresholds

class DefaultConfig:
    def __init__(self, model_name: Literal["cnn_b", "cnn", "lightgbm"], thresholds: dict, config_path: Path, weights_path: Path):
        if model_name == "cnn":
            raise NotImplementedError("The 'cnn' model is not implemented yet. Please use 'cnn_b' or 'lightgbm'.")
        
        self.config_path = config_path or self.get_default_config(model_name)
        self.weights_path = weights_path or self.get_default_weights(model_name)

        print(f"Using weights path: {self.weights_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} does not exist.")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file {self.weights_path} does not exist.")

        self.config = self.load_config(model_name, self.weights_path, thresholds)

    def get_default_config(self, model_name: str) -> Path:
        """Get the default config path based on model name."""
        model_dir = Path(__file__).parent / "model"
        default_configs = {
            "cnn_b":  "best_cnn_b_config.yaml",
            "cnn":  "cnn_config.yaml",
            "lightgbm":  "best_lgbm_config.yaml",
        }
        config_path = model_dir / default_configs.get(model_name)
        print(f"Using default config path: {config_path}")
        return config_path
        
    def get_default_weights(self, model_name: str) -> Path:
        """Get the default weights path based on model name."""
        model_dir = Path(__file__).parent / "model"
        default_weights = {
            "cnn_b":  "best_cnn_b.h5",
            "cnn":  "cnn_weights.h5",
            "lightgbm":  "best_lightgbm.pkl",
        }
        weights_path = model_dir / default_weights.get(model_name)
        print(f"Using default weights path: {weights_path}")
        return weights_path

    def load_config(self, model_name: str, weights_path: Path, thresholds: Thresholds) -> dict:
        """Load the configuration from the YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} does not exist.")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = {
            "cnn_b": CNN_B_Config,
            "cnn": None,  # Placeholder for future CNN config class
            "lightgbm": LIGHTGBM_Config,  # Placeholder for future LightGBM config class
        }

        config = model_config.get(model_name)(config, weights_path, thresholds)
        return config

    def get_config(self) -> dict:
        """Return the loaded configuration."""
        return self.config