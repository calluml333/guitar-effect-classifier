#!/usr/bin/env python3
"""Download, extract, and prepare the IDMT-SMT-Guitar dataset for local use.

The script is designed to keep all downloaded and processed data out of version
control. By default it writes to:

- data/downloads/idmt_smt_guitar/   (raw downloaded archives)
- data/raw/idmt_smt_guitar/        (selected WAV files for local training)

It can download the Zenodo record directly, or use a local archive path if one
already exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "data" / "downloads" / "idmt_smt_guitar"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "idmt_smt_guitar"
DEFAULT_RECORD_ID = "7544110"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the IDMT-SMT-Guitar dataset locally")
    parser.add_argument("--record-id", default=DEFAULT_RECORD_ID, help="Zenodo record ID")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Directory for downloaded archives")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for selected WAV files")
    parser.add_argument("--dataset-ids", default="2,3,4", help="Comma-separated dataset IDs to include (e.g. 2,3,4)")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[16, 24, 32], help="Target bit depth")
    parser.add_argument("--source-archive", default=None, help="Path to an existing local zip/tar.gz archive")
    parser.add_argument("--manifest", default=None, help="Path to a CSV manifest to read (skip discovery)")
    parser.add_argument("--write-manifest", default=None, help="Path to write a CSV manifest of discovered candidates")
    parser.add_argument("--skip-verify", action="store_true", help="Skip sample rate / bit depth verification when copying")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without downloading or copying files")
    parser.add_argument("--force-download", action="store_true", help="Re-download archives even if they already exist")
    return parser.parse_args()


def normalize_dataset_ids(raw_value: str) -> List[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_archive_name(entry: dict, download_url: str) -> str:
    for key in ("key", "filename", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    parsed_name = Path(download_url).name
    if parsed_name in {"content", "download"}:
        parts = [part for part in Path(download_url).parts if part not in {"", "api", "records", "files"}]
        if len(parts) >= 2:
            return parts[-2]
    return parsed_name


def format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
        size /= 1024.0


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def download_file_with_progress(download_url: str, destination_path: Path, description: str) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    downloaded_bytes = 0
    last_update = 0.0

    with urllib.request.urlopen(download_url) as response, destination_path.open("wb") as handle:
        total_size = response.getheader("Content-Length")
        total_bytes = int(total_size) if total_size and total_size.isdigit() else None
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded_bytes += len(chunk)
            now = time.perf_counter()
            if now - last_update >= 0.2 or (total_bytes and downloaded_bytes >= total_bytes):
                elapsed = max(now - start_time, 1e-6)
                speed = downloaded_bytes / elapsed if elapsed else 0.0
                if total_bytes:
                    percent = min(100.0, (downloaded_bytes / total_bytes) * 100.0)
                    bar_width = 20
                    filled = int(bar_width * percent / 100.0)
                    bar = "#" * filled + "-" * (bar_width - filled)
                    eta_seconds = (total_bytes - downloaded_bytes) / speed if speed > 0 else 0.0
                    print(
                        f"\r{description}: [{bar}] {percent:>5.1f}% {format_size(downloaded_bytes)}/{format_size(total_bytes)} "
                        f"{format_size(int(speed))}/s ETA {format_duration(eta_seconds)}",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r{description}: {format_size(downloaded_bytes)} downloaded in {format_duration(elapsed)} "
                        f"({format_size(int(speed))}/s)",
                        end="",
                        flush=True,
                    )
                last_update = now

    print()


def download_record_archives(record_id: str, target_dir: Path, force_download: bool) -> List[Path]:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url) as response:
        payload = json.load(response)

    files = payload.get("files", [])
    if not files:
        raise RuntimeError(f"No files were found for Zenodo record {record_id}")

    downloaded: List[Path] = []
    for entry in files:
        download_url = entry.get("links", {}).get("self")
        if not download_url:
            continue
        archive_name = resolve_archive_name(entry, download_url)
        archive_path = target_dir / archive_name
        if archive_path.exists() and not force_download:
            print(f"Archive already exists: {archive_path}")
            downloaded.append(archive_path)
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {archive_name} from {download_url}")
        download_file_with_progress(download_url, archive_path, f"  {archive_name}")
        downloaded.append(archive_path)
    return downloaded


def extract_archive(archive_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    if any(destination_dir.iterdir()):
        print(f"Extracted files already present in {destination_dir}")
        return destination_dir

    print(f"Extracting {archive_path.name} to {destination_dir}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as handle:
            handle.extractall(destination_dir)
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(destination_dir)
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path}")
    return destination_dir


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


def select_audio_files(extracted_root: Path, dataset_ids: Sequence[str], sample_rate: int, bit_depth: int, include_all: bool = False) -> List[Path]:
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


def write_manifest(output_dir: Path, copied_files: Sequence[Path], extracted_root: Path) -> None:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename"])
        writer.writeheader()
        for path in copied_files:
            writer.writerow({"filename": path.name})


def main() -> None:
    args = parse_args()
    dataset_ids = normalize_dataset_ids(args.dataset_ids)
    download_dir = Path(args.download_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run only. No files will be downloaded or copied.")
        print(f"Would use dataset IDs: {dataset_ids}")
        print(f"Would download to: {download_dir}")
        print(f"Would copy selected WAV files to: {output_dir}")
        return

    if args.source_archive:
        archive_path = Path(args.source_archive).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        archives = [archive_path]
    else:
        archives = download_record_archives(args.record_id, download_dir, args.force_download)

    extracted_roots: List[Path] = []
    for archive_path in archives:
        extracted_root = download_dir / archive_path.stem
        # If the extracted folder already exists and contains files, skip re-extraction
        if extracted_root.exists() and extracted_root.is_dir() and any(extracted_root.iterdir()):
            print(f"Skipping extraction; files already present in {extracted_root}")
            extracted_roots.append(extracted_root)
            continue
        if not archive_path.exists():
            print(f"Archive file not found, skipping: {archive_path}")
            continue
        extracted_roots.append(extract_archive(archive_path, extracted_root))
    candidates: List[dict] = []

    # If a manifest is provided, read it and use those paths
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                # Expect at least source_path or archive+relative_path
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
                    # try to resolve using extracted roots
                    archive_name = row["archive"]
                    matching_root = next((r for r in extracted_roots if r.name == archive_name), None)
                    if matching_root:
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
    else:
        for extracted_root in extracted_roots:
            # No manifest provided -> include all correctly formatted WAV files
            candidates.extend(select_audio_files(extracted_root, dataset_ids, args.sample_rate, args.bit_depth, include_all=True))

    # Optionally write a discovery manifest for curation
    if args.write_manifest:
        out_manifest = Path(args.write_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        with out_manifest.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["source_path", "archive", "relative_path", "filename", "samplerate", "bit_depth", "dataset_id"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for cand in candidates:
                writer.writerow({k: cand.get(k) for k in fieldnames})
        print(f"Wrote discovery manifest: {out_manifest}")

    if args.dry_run:
        print(f"Found {len(candidates)} candidate WAV files (dry-run)")
        return

    copied_files = copy_selected_files(candidates, output_dir, skip_verify=args.skip_verify) if candidates else []
    write_manifest(output_dir, copied_files, extracted_roots[0] if extracted_roots else output_dir)

    print(f"Prepared {len(copied_files)} WAV files in {output_dir}")
    print(f"Manifest written to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
