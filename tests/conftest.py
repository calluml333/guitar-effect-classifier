"""Pytest configuration and fixtures."""

import pytest
import torch
from pathlib import Path


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return torch.device("cpu")


@pytest.fixture
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a temporary audio file for testing."""
    import torchaudio

    # Generate 1 second of dummy audio at 16kHz
    sample_rate = 16000
    duration = 1.0
    waveform = torch.randn(1, int(sample_rate * duration))

    audio_path = tmp_path / "test_audio.wav"
    torchaudio.save(audio_path, waveform, sample_rate)

    return audio_path


@pytest.fixture(scope="session")
def config():
    """Return configuration for all tests."""
    from src.config import Config

    return Config()
