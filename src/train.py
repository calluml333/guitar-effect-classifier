"""Training and validation loop for the guitar effect classifier.

Usage:
    python -m src.train --manifest data/manifest.csv --feature hf --epochs 10
"""
import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from src import config
from src.dataset import GuitarEffectsDataset
from src.model import build_classifier_from_dataset_sample


def collate_fn(batch):
    xs, ys = zip(*batch)
    # xs can be tensors of different shapes for HF embeddings (1D) or spectrograms (2D)
    if xs[0].dim() == 1:
        x = torch.stack(xs)
    else:
        # flatten spectrograms
        x = torch.stack([xx.flatten() for xx in xs])
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    losses = []
    preds = []
    targets = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


def validate(model, loader, loss_fn, device, n_classes: int = None):
    """Run validation. `n_classes` fixes the label set used for precision/
    recall/confusion-matrix so their shape/order is stable across calls even
    if a given batch doesn't contain every class (otherwise sklearn infers
    the label set from whatever classes happen to appear).
    """
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            losses.append(loss.item())
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())
    labels = list(range(n_classes)) if n_classes is not None else None
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0, labels=labels)
    cm = confusion_matrix(targets, preds, labels=labels)
    return np.mean(losses), acc, prec, rec, f1, cm


def main(args):
    manifest = args.manifest
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dataset = GuitarEffectsDataset(manifest, sr=args.sr, duration=args.duration, feature=args.feature, hf_model_name=args.hf_model)
    # split
    n = len(dataset)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    split = int(n * args.val_split)
    train_idx, val_idx = idxs[split:], idxs[:split]
    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    # get sample to infer model size
    sample_x, _ = dataset[0]
    n_classes = len(dataset.label2idx)
    model = build_classifier_from_dataset_sample(sample_x, n_classes=n_classes)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx2label = {idx: label for label, idx in dataset.label2idx.items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc, prec, rec, f1, cm = validate(model, val_loader, loss_fn, device, n_classes=n_classes)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - {dt:.1f}s - train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} f1={f1:.4f}")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        )
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state": model.state_dict(),
                "label2idx": dataset.label2idx,
                "feature": args.feature,
            }
            if args.feature == "hf":
                checkpoint["hf_model_name"] = args.hf_model
            torch.save(checkpoint, out_dir / "best.pth")
            with open(out_dir / "confusion_matrix.json", "w") as f:
                json.dump({"labels": [idx2label[i] for i in range(len(idx2label))], "matrix": cm.tolist()}, f, indent=2)
    print("Training complete. Best val acc:", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--feature", type=str, default=config.DEFAULT_FEATURE, choices=["hf", "log-mel", "waveform"])
    parser.add_argument("--hf-model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--sr", type=int, default=config.SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=config.AUDIO_DURATION)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()
    main(args)
