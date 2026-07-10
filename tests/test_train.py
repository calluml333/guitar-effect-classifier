"""Tests for the training/validation loop in src/train.py."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import EffectClassifier
from src.train import collate_fn, train_one_epoch, validate

N_CLASSES = 3
INPUT_DIM = 4


class _TinyDataset(Dataset):
    """Small, linearly separable dataset: each class has a fixed feature
    pattern plus a little noise, so a model can actually learn to fit it."""

    def __init__(self, n_per_class: int = 4):
        labels = list(range(N_CLASSES)) * n_per_class
        class_centers = torch.eye(N_CLASSES, INPUT_DIM) * 3.0
        self.x = [class_centers[label] + 0.05 * torch.randn(INPUT_DIM) for label in labels]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class _TinySpectrogramDataset(Dataset):
    """Dataset returning 2-D 'spectrogram' tensors, to exercise the flatten path."""

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.randn(2, 3), idx % N_CLASSES


def test_collate_fn_stacks_1d_feature_tensors():
    batch = [(torch.randn(INPUT_DIM), 0), (torch.randn(INPUT_DIM), 1)]
    x, y = collate_fn(batch)
    assert x.shape == (2, INPUT_DIM)
    assert y.dtype == torch.long
    assert y.tolist() == [0, 1]


def test_collate_fn_flattens_2d_spectrogram_tensors():
    ds = _TinySpectrogramDataset()
    batch = [ds[i] for i in range(len(ds))]
    x, y = collate_fn(batch)
    assert x.shape == (4, 6)  # flattened 2x3 -> 6
    assert y.shape == (4,)


def test_train_one_epoch_returns_finite_loss_and_valid_accuracy():
    ds = _TinyDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = EffectClassifier(input_dim=INPUT_DIM, n_classes=N_CLASSES, hidden_dim=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, acc = train_one_epoch(model, loader, opt, loss_fn, torch.device("cpu"))

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0


def test_validate_returns_metrics_with_expected_shapes():
    ds = _TinyDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    model = EffectClassifier(input_dim=INPUT_DIM, n_classes=N_CLASSES, hidden_dim=8)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, acc, prec, rec, f1, cm = validate(model, loader, loss_fn, torch.device("cpu"))

    assert np.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert cm.shape == (N_CLASSES, N_CLASSES)


def test_training_reduces_loss_over_multiple_epochs():
    """A minimal end-to-end sanity check that the loop actually learns."""
    ds = _TinyDataset(n_per_class=8)
    loader = DataLoader(ds, batch_size=6, shuffle=True, collate_fn=collate_fn)
    model = EffectClassifier(input_dim=INPUT_DIM, n_classes=N_CLASSES, hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    first_loss, _ = train_one_epoch(model, loader, opt, loss_fn, torch.device("cpu"))
    for _ in range(10):
        last_loss, _ = train_one_epoch(model, loader, opt, loss_fn, torch.device("cpu"))

    assert last_loss < first_loss
