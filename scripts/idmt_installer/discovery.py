"""Finding and filtering candidate WAV files inside an extracted archive."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

import soundfile as sf


def is_dataset_match(path_name: str, dataset_ids: Sequence[str]) -> bool:
    lowered = path_name.lower()
    for dataset_id in dataset_ids:
        # match common forms like dataset2, dataset-2, dataset_2, d2
        patterns = [rf"dataset[_ -]?{dataset_id}", rf"d{dataset_id}"]
        if any(re.search(pattern, lowered) for pattern in patterns):
            return True
        # also match explicit audio folder paths like dataset2/audio
        if re.search(rf"dataset[_ -]?{dataset_id}(/|\\)audio", lowered):
            return True
    return False


def is_clean_candidate(path_name: str) -> bool:
    lowered = path_name.lower()
    excluded = ["distortion", "overdrive", "fuzz", "delay", "reverb", "chorus", "effect", "processed"]
    if any(token in lowered for token in excluded):
        return False
    included = ["clean", "dry", "direct", "noeffect", "no_effect", "original", "raw", "acoustic", "plain"]
    if any(token in lowered for token in included):
        return True
    return True


def select_audio_files(extracted_root: Path, dataset_ids: Sequence[str], sample_rate: int, bit_depth: int, include_all: bool = False) -> List[dict]:
    # Discover candidate files and return rich metadata for each
    candidates: List[dict] = []
    for wav_path in extracted_root.rglob("*.wav"):
        if not wav_path.is_file():
            continue
        rel_path = wav_path.relative_to(extracted_root).as_posix().lower()
        # When include_all is False, apply the dataset/name-based filters.
        if not include_all:
            if not is_dataset_match(rel_path, dataset_ids):
                continue
            # Special-case dataset 4: only include specific subfolders
            if "4" in dataset_ids and "dataset4" in rel_path:
                check_path = rel_path.replace("_", " ").replace("-", " ")
                allowed = ["dataset4/carrer sg", "dataset4/ibanez 2820"]
                if not any(a in check_path for a in allowed):
                    continue
            if not is_clean_candidate(rel_path):
                continue
        try:
            info = sf.info(str(wav_path))
        except Exception as exc:
            print(f"Skipping unreadable file {wav_path}: {exc}")
            continue
        candidate = {
            "source_path": str(wav_path.resolve()),
            "archive": extracted_root.name,
            "relative_path": wav_path.relative_to(extracted_root).as_posix(),
            "filename": wav_path.name,
            "samplerate": info.samplerate,
            "bit_depth": getattr(info, "subtype", "").upper(),
            "dataset_id": None,
        }
        # determine dataset id from path
        m = re.search(r"dataset[_ -]?(\d)|d(\d)", rel_path)
        if m:
            candidate["dataset_id"] = m.group(1) or m.group(2)
        candidates.append(candidate)
    return candidates


def sanitize_name(path_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(path_name).stem).strip("_")
    return f"{stem}.wav"
