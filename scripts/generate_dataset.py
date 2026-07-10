"""Generate a labeled dataset by applying DSP effects to clean guitar recordings.

Usage:
  python scripts/generate_dataset.py --input-dir data/raw --out-dir data/generated --manifest data/manifest.csv
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

# Ensure the repository root is on sys.path so `from src...` imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import librosa
import numpy as np
import soundfile as sf

from src import config
from src.effects import apply_effect_by_name
from src.utils import ensure_mono, format_duration

EFFECTS = config.EFFECT_CLASSES


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


def process_file(path: Path, out_dir: Path, sr: int, samples_per_input: int = 1):
    wave_np, orig_sr = sf.read(str(path))
    wave_np = ensure_mono(np.asarray(wave_np))
    if orig_sr != sr:
        wave_np = librosa.resample(wave_np.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    results = []
    for effect in EFFECTS:
        for i in range(samples_per_input):
            params = random_params_for(effect)
            processed = apply_effect_by_name(wave_np, sr, effect, params)
            label_dir = out_dir / effect
            label_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{path.stem}_{effect}_{i}.wav"
            out_path = label_dir / out_name
            # save using soundfile (writes numpy arrays)
            sf.write(str(out_path), processed.astype(np.float32), sr)
            try:
                rel_name = str(Path(out_path).resolve().relative_to(REPO_ROOT))
            except Exception:
                rel_name = str(out_path)
            results.append({"filename": rel_name, "label": effect, "params": json.dumps(params)})
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
    total_files = len(files)
    started = time.perf_counter()

    for idx, p in enumerate(files, start=1):
        rows.extend(process_file(p, out_dir, sr=args.sr, samples_per_input=args.samples_per_input))

        elapsed = max(time.perf_counter() - started, 1e-6)
        rate = idx / elapsed
        eta = (total_files - idx) / rate if rate > 0 else 0.0
        progress = idx / total_files
        bar_width = 24
        filled = int(progress * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\rGenerating: [{bar}] {idx}/{total_files} ({progress * 100:5.1f}%) "
            f"Elapsed {format_duration(elapsed)} ETA {format_duration(eta)}",
            end="",
            flush=True,
        )
    print()

    # write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "params"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    total_elapsed = time.perf_counter() - started
    generated_files = total_files * len(EFFECTS) * args.samples_per_input
    avg_files_per_sec = generated_files / total_elapsed if total_elapsed > 0 else 0.0
    print(f"Wrote manifest with {len(rows)} rows to {manifest_path}")
    print(
        "Generation summary: "
        f"inputs={total_files}, outputs={generated_files}, "
        f"elapsed={format_duration(total_elapsed)}, rate={avg_files_per_sec:.2f} files/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/generated")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--sr", type=int, default=config.SAMPLE_RATE)
    parser.add_argument("--samples-per-input", type=int, default=1, help="How many generated variants per input per effect")
    args = parser.parse_args()
    main(args)
