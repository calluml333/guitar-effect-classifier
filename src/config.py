"""Shared constants for the Guitar Effects Classification project.

Centralizes values that are duplicated across dataset generation, training,
and inference code so they can't drift out of sync with each other.
"""

SAMPLE_RATE = 32000  # Hz
AUDIO_DURATION = 3.0  # seconds; used to trim/pad clips for train/predict

EFFECT_CLASSES = [
    "clean",
    "overdrive",
    "distortion",
    "fuzz",
    "chorus",
    "delay",
    "reverb",
]

DEFAULT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEFAULT_FEATURE = "hf"
