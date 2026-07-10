"""Generate evaluation visualizations and reports for a trained checkpoint.

This is a manually-run analysis step, separate from training -- it doesn't
run automatically as part of `src.train`.

Produces, in --out-dir (default outputs/visualizations):
- loss_curve.png       : train/val loss per epoch (needs --history)
- accuracy_curve.png   : train/val accuracy per epoch (needs --history)
- confusion_matrix.png : heatmap, either recomputed fresh over --manifest or
                          (if --manifest is omitted) read from the checkpoint's
                          saved confusion_matrix.json
- predictions.csv      : per-file true/predicted label, confidence, and
                          top-k breakdown (only when --manifest is given)

Usage:
    python scripts/evaluate.py \
        --checkpoint models/best.pth \
        --manifest data/manifest.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src import config
from src.audio_processing import load_audio
from src.features import HFEmbedder
from src.predict import (
    build_model_from_feature,
    extract_feature,
    load_checkpoint,
    resolve_feature_settings,
)
from src.utils import print_progress


def plot_training_curves(history: List[Dict], out_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'loss_curve.png'}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax.plot(epochs, [h["val_acc"] for h in history], marker="o", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Training / Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'accuracy_curve.png'}")


def plot_confusion_matrix(matrix: np.ndarray, labels: List[str], out_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_dir / 'confusion_matrix.png'}")


def evaluate_manifest(
    manifest_path: str,
    checkpoint: Dict,
    sr: int,
    duration: float,
    topk: int,
    use_cuda: bool,
) -> "pd.DataFrame":
    """Run inference over every row of a manifest CSV and return a DataFrame
    of filename/true_label/predicted_label/confidence/correct/topk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    feature, hf_model_name = resolve_feature_settings(checkpoint)
    label2idx = checkpoint["label2idx"]
    idx2label = {int(idx): label for label, idx in label2idx.items()}

    if feature == "hf":
        print(f"Loading pretrained model '{hf_model_name}' -- may download weights on first run.", flush=True)
    embedder = HFEmbedder(model_name=hf_model_name, device="cpu") if feature == "hf" else None

    df = pd.read_csv(manifest_path)
    total = len(df)
    print(f"Evaluating {total} files from {manifest_path} (feature={feature})...", flush=True)
    model: Optional[torch.nn.Module] = None
    rows = []
    started = time.perf_counter()
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        wave = load_audio(row["filename"], sr=sr, duration=duration)
        feature_tensor = extract_feature(wave, sr, feature=feature, hf_model_name=hf_model_name, embedder=embedder)
        flat = feature_tensor.flatten().unsqueeze(0).to(device)
        if model is None:
            model = build_model_from_feature(feature_tensor, n_classes=len(label2idx), device=device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(flat).squeeze(0), dim=-1).cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:topk]
        top_predictions = {idx2label[int(i)]: round(float(probs[int(i)]), 4) for i in top_indices}
        predicted_label = idx2label[int(np.argmax(probs))]
        rows.append(
            {
                "filename": row["filename"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "confidence": float(np.max(probs)),
                "correct": predicted_label == row["label"],
                "top_predictions": json.dumps(top_predictions),
            }
        )
        running_acc = sum(r["correct"] for r in rows) / len(rows)
        print_progress("Evaluating", i, total, started, extra=f"acc={running_acc:.3f}")
    print()
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history) if args.history else Path(args.checkpoint).parent / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, out_dir)
    else:
        print(f"No training history found at {history_path}, skipping loss/accuracy curves.")

    checkpoint = load_checkpoint(args.checkpoint)
    label2idx = checkpoint["label2idx"]
    idx2label = {int(idx): label for label, idx in label2idx.items()}
    ordered_labels = [idx2label[i] for i in range(len(idx2label))]

    if args.manifest:
        results = evaluate_manifest(
            args.manifest, checkpoint, sr=args.sr, duration=args.duration, topk=args.topk, use_cuda=args.use_cuda
        )
        results.to_csv(out_dir / "predictions.csv", index=False)
        print(f"Wrote {out_dir / 'predictions.csv'} ({len(results)} rows)")

        accuracy = results["correct"].mean()
        print(f"Accuracy over {args.manifest}: {accuracy:.2%}")

        cm = confusion_matrix(results["true_label"], results["predicted_label"], labels=ordered_labels)
        plot_confusion_matrix(cm, ordered_labels, out_dir, title=f"Confusion Matrix ({Path(args.manifest).name})")

        n_examples = min(args.num_examples, len(results))
        correct = results[results["correct"]].head(n_examples)
        incorrect = results[~results["correct"]].head(n_examples)
        print(f"\nSample correct predictions ({len(correct)}):")
        print(correct[["filename", "true_label", "predicted_label", "confidence"]].to_string(index=False))
        print(f"\nSample incorrect predictions ({len(incorrect)}):")
        print(incorrect[["filename", "true_label", "predicted_label", "confidence"]].to_string(index=False))
    else:
        cm_path = Path(args.confusion_matrix) if args.confusion_matrix else Path(args.checkpoint).parent / "confusion_matrix.json"
        if cm_path.exists():
            with open(cm_path) as f:
                cm_data = json.load(f)
            plot_confusion_matrix(
                np.array(cm_data["matrix"]), cm_data["labels"], out_dir, title="Confusion Matrix (best training epoch)"
            )
        else:
            print(f"No --manifest given and no confusion matrix found at {cm_path}, skipping confusion matrix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default="models/best.pth")
    parser.add_argument("--manifest", type=str, default=None, help="Recompute confusion matrix + predictions.csv over this manifest")
    parser.add_argument("--history", type=str, default=None, help="Defaults to training_history.json next to --checkpoint")
    parser.add_argument("--confusion-matrix", type=str, default=None, help="Defaults to confusion_matrix.json next to --checkpoint (used only without --manifest)")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations")
    parser.add_argument("--sr", type=int, default=config.SAMPLE_RATE)
    parser.add_argument("--duration", type=float, default=config.AUDIO_DURATION)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num-examples", type=int, default=5, help="How many correct/incorrect examples to print")
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    main(args)
