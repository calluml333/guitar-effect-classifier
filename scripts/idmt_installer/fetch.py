"""Downloading and extracting IDMT-SMT-Guitar Zenodo archives."""
from __future__ import annotations

import json
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from src.utils import format_duration, format_size


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
