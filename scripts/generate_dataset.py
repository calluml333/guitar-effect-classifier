"""Generate a labeled dataset by applying DSP effects to clean guitar recordings.

Usage:
  python scripts/generate_dataset.py --input-dir data/raw --out-dir data/generated --manifest data/manifest.csv
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torchaudio

from src.effects import apply_effect_by_name

EFFECTS = ["clean", "overdrive", "distortion", "fuzz", "chorus", "delay", "reverb"]


def random_params_for(effect: str):
    if effect == "clean":
        return {}
    if effect == "overdrive":
        return {"gain": random.uniform(1.5, 6.0), "tone": random.uniform(0.3, 0.95)}
    if effect == "distortion":
        return {"drive": random.uniform(3.0, 12.0), "threshold": random.uniform(0.3, 0.9)}
    if effect == "fuzz":
        return {"gain": random.uniform(6.0, 20.0), "bias": random.uniform(-0.2, 0.2)}
    if effect == "chorus":
        return {"depth_ms": random.uniform(4.0, 25.0), "rate_hz": random.uniform(0.2, 1.5), "mix": random.uniform(0.2, 0.8)}
    if effect == "delay":
        return {"delay_ms": random.uniform(120.0, 700.0), "feedback": random.uniform(0.2, 0.6), "mix": random.uniform(0.2, 0.7)}
    if effect == "reverb":
        return {"ir_len_s": random.uniform(0.5, 2.5), "decay": random.uniform(1.0, 4.0), "mix": random.uniform(0.2, 0.8)}
    return {}


def ensure_mono(tensor):
    if tensor.ndim == 2:
        return tensor.mean(dim=0)
    return tensor


def process_file(path: Path, out_dir: Path, sr: int, samples_per_input: int = 1):
    waveform, orig_sr = torchaudio.load(str(path))
    waveform = ensure_mono(waveform)
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)
    wave_np = waveform.squeeze(0).numpy()
    results = []
    for effect in EFFECTS:
        for i in range(samples_per_input):
            params = random_params_for(effect)
            processed = apply_effect_by_name(wave_np, sr, effect, params)
            label_dir = out_dir / effect
            label_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{path.stem}_{effect}_{i}.wav"
            out_path = label_dir / out_name
            # save using torchaudio (expects tensor [channels, samples])
            tensor = torchaudio.functional.to_tensor(processed).float()
            torchaudio.save(str(out_path), tensor, sr)
            results.append({"filename": str(out_path.relative_to(Path.cwd())), "label": effect, "params": json.dumps(params)})
    return results


def main(args):
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    rows = []
    files = list(input_dir.glob("**/*.wav"))
    if not files:
        print(f"No WAV files found in {input_dir}. Place clean guitar WAVs there.")
        return
    for p in files:
        print(f"Processing {p}")
        rows.extend(process_file(p, out_dir, sr=args.sr, samples_per_input=args.samples_per_input))
    # write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "params"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote manifest with {len(rows)} rows to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/generated")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--sr", type=int, default=32000)
    parser.add_argument("--samples-per-input", type=int, default=1, help="How many generated variants per input per effect")
    args = parser.parse_args()
    main(args)
"""Placeholder: Generate synthetic guitar effects dataset."""

if __name__ == "__main__":
    pass
