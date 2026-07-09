import tempfile
from pathlib import Path

import torch

from src.audio_processing import load_audio, waveform_to_log_mel
from src.dataset import GuitarEffectsDataset
from src.model import EffectClassifier, build_classifier_from_dataset_sample
from src.predict import (
    build_model_from_feature,
    extract_feature,
    format_predictions,
    load_checkpoint,
)


def test_load_audio_and_log_mel():
    # generate a short sine wave and write to a temp file
    sr = 16000
    t = torch.linspace(0, 1.0, int(sr * 1.0))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    tmpdir = tempfile.TemporaryDirectory()
    p = Path(tmpdir.name) / "test.wav"
    import soundfile as sf
    sf.write(str(p), wave.numpy(), sr)
    loaded = load_audio(str(p), sr=sr, duration=1.0)
    assert isinstance(loaded, torch.Tensor)
    mel = waveform_to_log_mel(loaded, sr=sr)
    assert mel.ndim == 2


def test_dataset_manifest_smoke(tmp_path):
    # create a fake manifest with one example
    wav_path = tmp_path / "example.wav"
    sr = 16000
    t = torch.linspace(0, 1.0, int(sr * 1.0))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    import soundfile as sf
    sf.write(str(wav_path), wave.numpy(), sr)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,label,params\n")
        f.write(f"{wav_path.as_posix()},clean,{{}}\n")
    ds = GuitarEffectsDataset(str(manifest), sr=sr, duration=1.0, feature="log-mel")
    x, y = ds[0]
    assert x.ndim == 2
    assert isinstance(y, int)


def test_effect_classifier_forward_shape():
    model = EffectClassifier(input_dim=16, n_classes=4)
    x = torch.randn(3, 16)
    logits = model(x)
    assert logits.shape == (3, 4)


def test_build_classifier_from_dataset_sample_infers_input_dim():
    sample = torch.randn(2, 3, 4)
    model = build_classifier_from_dataset_sample(sample, n_classes=7)
    assert isinstance(model, EffectClassifier)


def test_format_predictions_formats_scores():
    predictions = [("clean", 0.5), ("distortion", 0.25)]
    text = format_predictions(predictions)
    assert "clean: 50.00%" in text
    assert "distortion: 25.00%" in text


def test_load_audio_pads_short_waveforms():
    sr = 16000
    short_wave = torch.zeros(1000, dtype=torch.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        import soundfile as sf
        sf.write(tmp.name, short_wave.numpy(), sr)
        loaded = load_audio(tmp.name, sr=sr, duration=1.0)
    assert loaded.shape[0] == sr


def test_dataset_builds_label_mapping(tmp_path):
    wav_path = tmp_path / "example.wav"
    sr = 16000
    t = torch.linspace(0, 1.0, int(sr * 1.0))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    import soundfile as sf
    sf.write(str(wav_path), wave.numpy(), sr)
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w") as f:
        f.write("filename,label,params\n")
        f.write(f"{wav_path.as_posix()},distortion,{{}}\n")
        f.write(f"{wav_path.as_posix()},clean,{{}}\n")
    ds = GuitarEffectsDataset(str(manifest), sr=sr, duration=1.0, feature="waveform")
    assert ds.label2idx["clean"] == 0
    assert ds.label2idx["distortion"] == 1


def test_load_checkpoint_reads_saved_state(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"model_state": {"layer.weight": torch.ones(1, 1)}, "label2idx": {"clean": 0, "distortion": 1}}, checkpoint_path)
    checkpoint = load_checkpoint(str(checkpoint_path))
    assert checkpoint["label2idx"]["clean"] == 0
    assert "model_state" in checkpoint


def test_build_model_from_feature_uses_feature_dimension():
    feature_tensor = torch.randn(4, 8)
    model = build_model_from_feature(feature_tensor, n_classes=3)
    assert isinstance(model, EffectClassifier)
    assert model.net[0].in_features == 32


def test_extract_feature_rejects_unknown_feature_type():
    wave = torch.randn(16000)
    try:
        extract_feature(wave, sr=16000, feature="unknown")
    except ValueError as exc:
        assert "Unsupported feature type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported feature type")


def test_load_checkpoint_requires_expected_keys(tmp_path):
    checkpoint_path = tmp_path / "bad_checkpoint.pth"
    torch.save({"model_state": {}}, checkpoint_path)
    try:
        load_checkpoint(str(checkpoint_path))
    except KeyError as exc:
        assert "label2idx" in str(exc)
    else:
        raise AssertionError("Expected KeyError for missing label2idx")
