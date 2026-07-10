"""Tests for checkpoint-driven feature-type auto-detection in src/predict.py."""
import tempfile
from pathlib import Path

import soundfile as sf
import torch

from src import config
from src.model import EffectClassifier
from src.predict import predict_audio, resolve_feature_settings


def test_resolve_feature_settings_uses_checkpoint_when_no_override():
    checkpoint = {"feature": "log-mel", "hf_model_name": "some/model"}
    feature, hf_model = resolve_feature_settings(checkpoint)
    assert feature == "log-mel"
    assert hf_model == "some/model"


def test_resolve_feature_settings_explicit_override_wins():
    checkpoint = {"feature": "log-mel", "hf_model_name": "some/model"}
    feature, hf_model = resolve_feature_settings(checkpoint, feature="waveform", hf_model_name="other/model")
    assert feature == "waveform"
    assert hf_model == "other/model"


def test_resolve_feature_settings_falls_back_to_config_defaults_for_old_checkpoints():
    checkpoint = {}  # no "feature"/"hf_model_name" keys, e.g. a pre-existing checkpoint
    feature, hf_model = resolve_feature_settings(checkpoint)
    assert feature == config.DEFAULT_FEATURE
    assert hf_model == config.DEFAULT_MODEL


def _make_sine_wav(path: Path, sr: int = 16000, duration: float = 1.0) -> None:
    t = torch.linspace(0, duration, int(sr * duration))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    sf.write(str(path), wave.numpy(), sr)


def test_predict_audio_auto_detects_feature_from_checkpoint(tmp_path):
    sr, duration = 16000, 1.0
    wav_path = tmp_path / "clip.wav"
    _make_sine_wav(wav_path, sr=sr, duration=duration)

    # A checkpoint trained with the "waveform" feature: input_dim == sr * duration samples.
    input_dim = int(sr * duration)
    label2idx = {"clean": 0, "overdrive": 1}
    model = EffectClassifier(input_dim=input_dim, n_classes=len(label2idx))
    checkpoint_path = tmp_path / "ckpt.pth"
    torch.save(
        {"model_state": model.state_dict(), "label2idx": label2idx, "feature": "waveform"},
        checkpoint_path,
    )

    # No `feature` argument passed -> must be auto-detected from the checkpoint as "waveform".
    preds = predict_audio(
        audio_path=str(wav_path),
        checkpoint_path=str(checkpoint_path),
        sr=sr,
        duration=duration,
        topk=2,
    )
    assert {label for label, _ in preds} == set(label2idx)


def test_predict_audio_explicit_feature_overrides_checkpoint(tmp_path):
    sr, duration = 16000, 1.0
    wav_path = tmp_path / "clip.wav"
    _make_sine_wav(wav_path, sr=sr, duration=duration)

    # Checkpoint recorded as "log-mel" (64 mel bands x time frames), but we'll
    # override to "waveform" at predict time and expect that to take effect.
    input_dim = int(sr * duration)
    label2idx = {"clean": 0, "overdrive": 1}
    model = EffectClassifier(input_dim=input_dim, n_classes=len(label2idx))
    checkpoint_path = tmp_path / "ckpt.pth"
    torch.save(
        {"model_state": model.state_dict(), "label2idx": label2idx, "feature": "log-mel"},
        checkpoint_path,
    )

    preds = predict_audio(
        audio_path=str(wav_path),
        checkpoint_path=str(checkpoint_path),
        feature="waveform",
        sr=sr,
        duration=duration,
        topk=2,
    )
    assert {label for label, _ in preds} == set(label2idx)
