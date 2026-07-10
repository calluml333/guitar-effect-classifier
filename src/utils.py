"""Small helpers shared across dataset generation, DSP, and CLI scripts."""

import time

import numpy as np


def ensure_mono(wave: np.ndarray) -> np.ndarray:
    """Average a (samples, channels) array down to mono (samples,).

    Arrays are already mono are returned unchanged. Assumes the
    `soundfile`-style (samples, channels) layout.
    """
    if wave.ndim == 2:
        return wave.mean(axis=-1)
    return wave


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1h02m03s', '2m03s', or '3s'."""
    seconds = max(int(seconds), 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def format_size(size_bytes: float) -> str:
    """Format a byte count as e.g. '512B', '1.5MB', '2.0GB'."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}{unit}"
        size /= 1024.0


def print_progress(prefix: str, current: int, total: int, started: float, extra: str = "") -> None:
    """Print an in-place progress bar (overwritten via carriage return).

    Call once per completed unit of work with a shared `started =
    time.perf_counter()` timestamp; call bare `print()` after the loop
    finishes to move off the in-place line. `extra` is appended after the
    percentage, e.g. a running loss or accuracy figure.
    """
    elapsed = max(time.perf_counter() - started, 1e-6)
    rate = current / elapsed
    eta = (total - current) / rate if rate > 0 else 0.0
    progress = current / total if total else 1.0
    bar_width = 24
    filled = int(progress * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)
    suffix = f" {extra}" if extra else ""
    print(
        f"\r{prefix}: [{bar}] {current}/{total} ({progress * 100:5.1f}%){suffix} "
        f"Elapsed {format_duration(elapsed)} ETA {format_duration(eta)}",
        end="",
        flush=True,
    )
