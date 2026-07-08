"""Inference routines for guitar effect classification."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

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


def extract_feature(wave: torch.Tensor, sr: int, feature: str = "hf", hf_model_name: str = "facebook/wav2vec2-base-960h") -> torch.Tensor:
    if feature == "waveform":
        return wave
    if feature == "log-mel":
        return waveform_to_log_mel(wave, sr=sr)
    if feature == "hf":
        embedder = HFEmbedder(model_name=hf_model_name, device="cpu")
        return embedder.extract(wave, sr=sr)
    raise ValueError(f"Unsupported feature type: {feature}")


def predict_audio(
    audio_path: str,
    checkpoint_path: str,
    feature: str = "hf",
    hf_model_name: str = "facebook/wav2vec2-base-960h",
    sr: int = 32000,
    duration: float = 3.0,
    topk: int = 3,
    use_cuda: bool = False,
) -> List[Tuple[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    label2idx = checkpoint["label2idx"]
    idx2label = {int(idx): label for label, idx in label2idx.items()}
    wave = load_audio(audio_path, sr=sr, duration=duration)
    feature_tensor = extract_feature(wave, sr, feature=feature, hf_model_name=hf_model_name)
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
