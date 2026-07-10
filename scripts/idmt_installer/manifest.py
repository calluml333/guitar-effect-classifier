"""Reading/writing CSV manifests of candidate and copied WAV files."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence

import soundfile as sf

from .discovery import sanitize_name


def read_candidates_manifest(manifest_path: Path, extracted_roots: Sequence[Path]) -> List[dict]:
    """Read a previously curated discovery manifest (see write_discovery_manifest).

    Each row either names a `source_path` directly, or an `archive` +
    `relative_path` to resolve against one of `extracted_roots` (expanding
    directories into individual WAV entries).
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    candidates: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("source_path"):
                candidates.append({
                    "source_path": row.get("source_path"),
                    "archive": row.get("archive"),
                    "relative_path": row.get("relative_path"),
                    "filename": row.get("filename") or Path(row.get("source_path")).name,
                    "samplerate": int(row["samplerate"]) if row.get("samplerate") else None,
                    "bit_depth": row.get("bit_depth"),
                    "dataset_id": row.get("dataset_id"),
                })
            elif row.get("archive") and row.get("relative_path"):
                archive_name = row["archive"]
                matching_root = next((r for r in extracted_roots if r.name == archive_name), None)
                if not matching_root:
                    continue
                src = matching_root / row["relative_path"]
                if src.exists() and src.is_dir():
                    # expand directory into individual WAV entries
                    for wav in src.rglob("*.wav"):
                        candidates.append({
                            "source_path": str(wav.resolve()),
                            "archive": archive_name,
                            "relative_path": wav.relative_to(matching_root).as_posix(),
                            "filename": wav.name,
                            "samplerate": int(row["samplerate"]) if row.get("samplerate") else None,
                            "bit_depth": row.get("bit_depth"),
                            "dataset_id": row.get("dataset_id"),
                        })
                else:
                    candidates.append({
                        "source_path": str(src.resolve()),
                        "archive": archive_name,
                        "relative_path": row["relative_path"],
                        "filename": Path(row["relative_path"]).name,
                        "samplerate": int(row["samplerate"]) if row.get("samplerate") else None,
                        "bit_depth": row.get("bit_depth"),
                        "dataset_id": row.get("dataset_id"),
                    })
    return candidates


def write_discovery_manifest(out_manifest: Path, candidates: Iterable[dict]) -> None:
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_path", "archive", "relative_path", "filename", "samplerate", "bit_depth", "dataset_id"]
    with out_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            writer.writerow({k: cand.get(k) for k in fieldnames})


def copy_selected_files(candidates: Iterable[dict], output_dir: Path, skip_verify: bool = False) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    used_names = set()
    for cand in candidates:
        src = Path(cand.get("source_path"))
        if not src.exists():
            print(f"Source file missing, skipping: {src}")
            continue
        candidate = sanitize_name(cand.get("filename", src.name))
        destination = output_dir / candidate
        index = 1
        while destination.exists() or candidate in used_names:
            suffix = f"_{index}"
            candidate = sanitize_name(f"{Path(candidate).stem}{suffix}.wav")
            destination = output_dir / candidate
            index += 1
        used_names.add(candidate)
        if not skip_verify:
            try:
                info = sf.info(str(src))
                # verify sample rate and bit depth if provided
                if "samplerate" in cand and cand["samplerate"] and info.samplerate != cand["samplerate"]:
                    print(f"Skipping {src}: samplerate {info.samplerate} != expected {cand['samplerate']}")
                    continue
            except Exception:
                print(f"Unable to verify {src}; copying anyway")
        shutil.copy2(src, destination)
        copied.append(destination)
    return copied


def write_manifest(output_dir: Path, copied_files: Sequence[Path]) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename"])
        writer.writeheader()
        for path in copied_files:
            writer.writerow({"filename": path.name})
