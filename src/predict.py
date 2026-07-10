"""Inference routines for guitar effect classification."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src import config
from src.audio_processing import load_audio, waveform_to_log_mel
from src.features import HFEmbedder
from src.model import EffectClassifier


def load_checkpoint(checkpoint_path: str, device: Optional[torch.device] = None) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state'.")
    if "label2idx" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'label2idx'.")
    return checkpoint


def build_model_from_feature(feature_tensor: torch.Tensor, n_classes: int, device: Optional[torch.device] = None) -> EffectClassifier:
    if feature_tensor.dim() > 1:
        input_dim = feature_tensor.numel()
    else:
        input_dim = feature_tensor.shape[0]
    model = EffectClassifier(input_dim=input_dim, n_classes=n_classes)
    if device is not None:
        model.to(device)
    return model


def resolve_feature_settings(
    checkpoint: Dict, feature: Optional[str] = None, hf_model_name: Optional[str] = None
) -> Tuple[str, str]:
    """Resolve the feature type/HF model to use, preferring explicit overrides,
    then whatever the checkpoint was trained with, then the project defaults.
    """
    resolved_feature = feature or checkpoint.get("feature") or config.DEFAULT_FEATURE
    resolved_hf_model = hf_model_name or checkpoint.get("hf_model_name") or config.DEFAULT_HF_MODEL
    return resolved_feature, resolved_hf_model


def extract_feature(
    wave: torch.Tensor,
    sr: int,
    feature: str = config.DEFAULT_FEATURE,
    hf_model_name: str = config.DEFAULT_HF_MODEL,
    embedder: Optional[HFEmbedder] = None,
) -> torch.Tensor:
    """Extract the given feature type from a waveform.

    For feature=='hf', pass a pre-built `embedder` when calling this in a
    loop over many files to avoid reloading the pretrained model each time.
    """
    if feature == "waveform":
        return wave
    if feature == "log-mel":
        return waveform_to_log_mel(wave, sr=sr)
    if feature == "hf":
        embedder = embedder or HFEmbedder(model_name=hf_model_name, device="cpu")
        return embedder.extract(wave, sr=sr)
    raise ValueError(f"Unsupported feature type: {feature}")


def predict_audio(
    audio_path: str,
    checkpoint_path: str,
    feature: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    sr: int = config.SAMPLE_RATE,
    duration: float = config.AUDIO_DURATION,
    topk: int = 3,
    use_cuda: bool = False,
) -> List[Tuple[str, float]]:
    """Run inference on an audio file.

    `feature`/`hf_model_name` default to whatever the checkpoint was trained
    with (see `resolve_feature_settings`); pass them explicitly to override.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    label2idx = checkpoint["label2idx"]
    idx2label = {int(idx): label for label, idx in label2idx.items()}
    resolved_feature, resolved_hf_model = resolve_feature_settings(checkpoint, feature, hf_model_name)
    wave = load_audio(audio_path, sr=sr, duration=duration)
    feature_tensor = extract_feature(wave, sr, feature=resolved_feature, hf_model_name=resolved_hf_model)
    flat = feature_tensor.flatten().unsqueeze(0).to(device)
    model = build_model_from_feature(feature_tensor, n_classes=len(label2idx), device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(flat)
        probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:topk]
    return [(idx2label[int(i)], float(probs[int(i)])) for i in top_indices]


def format_predictions(predictions: List[Tuple[str, float]]) -> str:
    lines = [f"{label}: {score:.2%}" for label, score in predictions]
    return "\n".join(lines)
