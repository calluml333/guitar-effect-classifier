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

DEFAULT_HF_MODEL = "facebook/wav2vec2-base-960h"
DEFAULT_FEATURE = "hf"
