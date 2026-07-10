"""Tests for scripts/evaluate.py's plotting and manifest-evaluation logic."""
import csv
import json

import numpy as np
import torch
import soundfile as sf

from scripts.evaluate import evaluate_manifest, plot_confusion_matrix, plot_training_curves
from src.model import EffectClassifier


def test_plot_training_curves_writes_both_png_files(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 2.0, "train_acc": 0.1, "val_loss": 1.9, "val_acc": 0.15},
        {"epoch": 2, "train_loss": 1.5, "train_acc": 0.3, "val_loss": 1.6, "val_acc": 0.25},
    ]
    plot_training_curves(history, tmp_path)
    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "accuracy_curve.png").exists()
    assert (tmp_path / "loss_curve.png").stat().st_size > 0
    assert (tmp_path / "accuracy_curve.png").stat().st_size > 0


def test_plot_confusion_matrix_writes_png(tmp_path):
    matrix = np.array([[3, 1], [0, 4]])
    plot_confusion_matrix(matrix, ["clean", "overdrive"], tmp_path, title="test")
    out_path = tmp_path / "confusion_matrix.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def _make_sine_wav(path, sr=16000, duration=1.0):
    t = torch.linspace(0, duration, int(sr * duration))
    wave = 0.1 * torch.sin(2 * 440.0 * 2 * 3.14159 * t)
    sf.write(str(path), wave.numpy(), sr)


def test_evaluate_manifest_scores_predictions_against_true_labels(tmp_path):
    sr, duration = 16000, 1.0
    input_dim = int(sr * duration)
    label2idx = {"clean": 0, "overdrive": 1}

    # a manifest of two files, one per class
    wav_paths = []
    for i, label in enumerate(label2idx):
        wav_path = tmp_path / f"clip_{i}.wav"
        _make_sine_wav(wav_path, sr=sr, duration=duration)
        wav_paths.append((str(wav_path), label))

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "params"])
        for path, label in wav_paths:
            writer.writerow([path, label, "{}"])

    model = EffectClassifier(input_dim=input_dim, n_classes=len(label2idx))
    checkpoint = {"model_state": model.state_dict(), "label2idx": label2idx, "feature": "waveform"}

    results = evaluate_manifest(
        str(manifest_path), checkpoint, sr=sr, duration=duration, topk=2, use_cuda=False
    )

    assert len(results) == 2
    assert set(results.columns) == {
        "filename", "true_label", "predicted_label", "confidence", "correct", "top_predictions"
    }
    assert set(results["true_label"]) == {"clean", "overdrive"}
    assert results["predicted_label"].isin(["clean", "overdrive"]).all()
    assert (results["correct"] == (results["true_label"] == results["predicted_label"])).all()
    # top_predictions should be valid JSON mapping label -> probability
    top_preds = json.loads(results.iloc[0]["top_predictions"])
    assert set(top_preds) == {"clean", "overdrive"}
