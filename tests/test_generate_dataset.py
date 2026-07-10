"""Tests for the output-size estimate in scripts/generate_dataset.py."""
import numpy as np
import soundfile as sf

from scripts.generate_dataset import (
    BYTES_PER_SAMPLE,
    WAV_HEADER_BYTES,
    estimate_output_bytes,
    EFFECTS,
)


def test_estimate_output_bytes_matches_actual_generated_size(tmp_path):
    sr = 16000
    duration_s = 2.0
    samples_per_input = 3

    wav_path = tmp_path / "clip.wav"
    t = np.arange(int(sr * duration_s)).astype(np.float32) / sr
    wave = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(wav_path), wave, sr)

    estimated = estimate_output_bytes([wav_path], sr=sr, samples_per_input=samples_per_input)

    expected_per_file = int(duration_s * sr) * BYTES_PER_SAMPLE + WAV_HEADER_BYTES
    expected_total = expected_per_file * len(EFFECTS) * samples_per_input
    assert estimated == expected_total


def test_estimate_output_bytes_scales_with_input_count_and_samples_per_input(tmp_path):
    sr = 16000
    t = np.arange(sr).astype(np.float32) / sr  # 1 second
    wave = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    wav_path = tmp_path / "one_second.wav"
    sf.write(str(wav_path), wave, sr)

    single = estimate_output_bytes([wav_path], sr=sr, samples_per_input=1)
    doubled_inputs = estimate_output_bytes([wav_path, wav_path], sr=sr, samples_per_input=1)
    doubled_samples = estimate_output_bytes([wav_path], sr=sr, samples_per_input=2)

    assert doubled_inputs == 2 * single
    assert doubled_samples == 2 * single


def test_estimate_output_bytes_empty_files_list_is_zero():
    assert estimate_output_bytes([], sr=16000, samples_per_input=1) == 0
