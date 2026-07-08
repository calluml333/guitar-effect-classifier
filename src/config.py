"""Configuration for Guitar Effects Classification System.

This module centralizes hyperparameters, paths, and model configurations
for reproducible experiments and easy hyperparameter tuning.
"""

from pathlib import Path
from typing import List
import yaml

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "training_logs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"

# Create directories if they don't exist
for dir_path in [CHECKPOINTS_DIR, LOGS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET & AUDIO
# ============================================================================

class Config:
    """Configuration class for guitar effects classification."""

    # Basic settings
    PROJECT_NAME: str = "guitar-effect-classifier"
    DESCRIPTION: str = "Audio classification for guitar effects"
    VERSION: str = "0.1.0"

    # Audio settings
    SAMPLE_RATE: int = 16000  # Hz
    AUDIO_DURATION: float = 10.0  # seconds
    AUDIO_LENGTH: int = int(SAMPLE_RATE * AUDIO_DURATION)  # samples
    MONO: bool = True
    NORMALIZE_AUDIO: bool = True

    # Effect classes
    EFFECT_CLASSES: List[str] = [
        "clean",
        "overdrive",
        "distortion",
        "fuzz",
        "chorus",
        "delay",
        "reverb",
    ]
    NUM_CLASSES: int = len(EFFECT_CLASSES)
    CLASS_TO_IDX: dict = {label: idx for idx, label in enumerate(EFFECT_CLASSES)}
    IDX_TO_CLASS: dict = {idx: label for label, idx in CLASS_TO_IDX.items()}

    # Dataset generation settings
    NUM_SAMPLES_PER_CLASS: int = 100  # Total: 700 samples
    GENERATE_AUGMENTED: bool = True
    AUGMENTATION_FACTOR: int = 1  # How many augmented versions per original

    # Effect parameter ranges (for synthetic generation)
    EFFECT_PARAMS: dict = {
        "overdrive": {
            "gain": (2.0, 5.0),  # dB
            "tone": (0.3, 0.7),  # Filter cutoff normalized
            "wet": (1.0, 1.0),  # Blend with dry
        },
        "distortion": {
            "gain": (5.0, 15.0),
            "tone": (0.2, 0.6),
            "wet": (1.0, 1.0),
        },
        "fuzz": {
            "gain": (15.0, 30.0),
            "tone": (0.1, 0.4),
            "wet": (1.0, 1.0),
        },
        "chorus": {
            "rate": (0.5, 2.0),  # Hz
            "depth": (0.005, 0.02),  # seconds
            "wet": (0.5, 0.8),
        },
        "delay": {
            "delay_time": (0.3, 0.6),  # seconds
            "feedback": (0.3, 0.7),
            "wet": (0.4, 0.7),
        },
        "reverb": {
            "room_size": (0.5, 0.9),
            "damping": (0.3, 0.7),
            "wet": (0.3, 0.6),
        },
    }

    # ========================================================================
    # MODEL & TRAINING
    # ========================================================================

    # Feature extraction model
    PRETRAINED_MODEL: str = "facebook/beats"  # or "facebook/ast-finetuned-audioset-10-10-0.9593"
    MODEL_TYPE: str = "beats"  # "beats" or "ast"
    FREEZE_FEATURE_EXTRACTOR: bool = False  # False = light finetuning
    EMBEDDING_DIM: int = 768  # BEATs uses 768-dim embeddings

    # Classifier architecture
    CLASSIFIER_HIDDEN_DIM: int = 256
    CLASSIFIER_NUM_LAYERS: int = 2
    DROPOUT_RATE: float = 0.3

    # Training hyperparameters
    BATCH_SIZE: int = 32
    VAL_BATCH_SIZE: int = 64
    NUM_EPOCHS: int = 30
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    GRADIENT_CLIP_MAX_NORM: float | None = 1.0

    # Optimizer
    OPTIMIZER: str = "adam"  # "adam" or "adamw"
    LR_SCHEDULER: str = "cosine"  # "cosine" or "linear"

    # Early stopping & checkpointing
    PATIENCE: int = 5  # Stop after 5 epochs no improvement
    SAVE_BEST_ONLY: bool = True
    CHECKPOINT_INTERVAL: int = 1  # Save every epoch

    # ========================================================================
    # EVALUATION & INFERENCE
    # ========================================================================

    CONFIDENCE_THRESHOLD: float = 0.5  # For binary decision
    SHOW_TOP_K: int = 3  # Show top-3 predictions
    USE_TEST_TIME_AUGMENTATION: bool = False

    # ========================================================================
    # LOGGING & REPRODUCIBILITY
    # ========================================================================

    RANDOM_SEED: int = 42
    NUM_WORKERS: int = 4  # DataLoader workers
    DEVICE: str = "cuda"  # or "cpu"

    # Logging
    LOG_INTERVAL: int = 10  # Log every N batches
    VERBOSE: bool = True

    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict))
        }

    @classmethod
    def save_yaml(cls, path: Path) -> None:
        """Save config as YAML."""
        with open(path, "w") as f:
            yaml.dump(cls.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: Path) -> None:
        """Load config from YAML."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)


if __name__ == "__main__":
    print("Guitar Effects Classifier Configuration")
    print("=" * 50)
    for key, value in Config.to_dict().items():
        print(f"{key}: {value}")
