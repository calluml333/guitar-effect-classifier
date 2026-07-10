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
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.idmt_installer import discovery, fetch, manifest  # noqa: E402

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
        archives = fetch.download_record_archives(args.record_id, download_dir, args.force_download)

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
        extracted_roots.append(fetch.extract_archive(archive_path, extracted_root))

    if args.manifest:
        candidates = manifest.read_candidates_manifest(Path(args.manifest), extracted_roots)
    else:
        candidates = []
        for extracted_root in extracted_roots:
            # No manifest provided -> include all correctly formatted WAV files
            candidates.extend(
                discovery.select_audio_files(
                    extracted_root, dataset_ids, args.sample_rate, args.bit_depth, include_all=True
                )
            )

    if args.write_manifest:
        out_manifest = Path(args.write_manifest)
        manifest.write_discovery_manifest(out_manifest, candidates)
        print(f"Wrote discovery manifest: {out_manifest}")

    copied_files = manifest.copy_selected_files(candidates, output_dir, skip_verify=args.skip_verify) if candidates else []
    manifest.write_manifest(output_dir, copied_files)

    print(f"Prepared {len(copied_files)} WAV files in {output_dir}")
    print(f"Manifest written to {output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
