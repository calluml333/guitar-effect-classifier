"""Audio preprocessing utilities.

Utilities for loading audio, resampling, mono conversion, normalization,
trim/pad to fixed duration, and generating log-mel spectrograms.

Functions return PyTorch tensors (float32).
"""
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch


def load_audio(
    path: str,
    sr: int = 32000,
    mono: bool = True,
    normalize: bool = True,
    trim_db: Optional[float] = 60.0,
    duration: Optional[float] = 3.0,
) -> torch.Tensor:
    """Load an audio file and apply standard preprocessing.

    Args:
        path: Path to audio file.
        sr: Target sample rate.
        mono: Convert to mono by averaging channels.
        normalize: Scale waveform to max abs 0.98.
        trim_db: If set, trim leading/trailing silence below
				 (max_db - trim_db).
        duration: If set, pad or trim to `duration` seconds.

    Returns:
        1-D `torch.Tensor` of shape (samples,) dtype float32.
    """
    data, orig_sr = sf.read(path, dtype="float32")
    # data shape: (samples,) or (samples, channels)
    if data.ndim == 2:
        # convert to mono by averaging channels if requested
        if mono:
            data = data.mean(axis=1)
        else:
            # keep channels-first format
            data = data.T
    # resample if needed
    if orig_sr != sr:
        if data.ndim == 1:
            data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
        else:
            data = np.stack([librosa.resample(d, orig_sr=orig_sr, target_sr=sr) for d in data], axis=0)  # noqa: E501
    # convert to torch tensor
    wave = torch.from_numpy(data).to(torch.float32)
    if wave.dim() == 2 and wave.shape[0] > 1 and mono:
        wave = wave.mean(dim=0)

    if trim_db is not None:
        # simple energy-based trim: compute frame-wise dB and find
		# region above threshold
        abs_wave = wave.abs()
        # compute short-time energy with a small frame (hop ~ 10 ms)
        hop = max(1, int(0.01 * sr))
        frame_len = hop * 2
        frames = (
            abs_wave.unfold(0, frame_len, hop).pow(2).mean(dim=1).clamp(min=1e-12)  # noqa: E501
        )
        db = 10.0 * torch.log10(frames)
        max_db = db.max()
        mask = db >= (max_db - trim_db)
        if mask.any():
            first = int(max(0, (mask.nonzero()[0].item() * hop) - frame_len))
            last = int(min(wave.shape[0], (mask.nonzero()[-1].item() * hop) + frame_len))  # noqa: E501
            wave = wave[first:last]

    if duration is not None:
        target_len = int(sr * duration)
        if wave.shape[0] > target_len:
            # center crop
            start = (wave.shape[0] - target_len) // 2
            wave = wave[start : start + target_len]
        elif wave.shape[0] < target_len:
            pad = target_len - wave.shape[0]
            wave = torch.nn.functional.pad(wave, (0, pad))

    if normalize:
        peak = wave.abs().max().clamp(min=1e-9)
        wave = wave / peak * 0.98

    return wave.to(torch.float32)


def waveform_to_log_mel(
    wave: torch.Tensor,
    sr: int = 32000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 320,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Convert a waveform tensor to a log-mel spectrogram (dB).

    Returns a tensor of shape (n_mels, time_frames).
    """
    if f_max is None:
        f_max = float(sr // 2)
    # ensure 1D numpy array
    if wave.dim() == 2:
        wave_np = wave.mean(dim=0).cpu().numpy()
    else:
        wave_np = wave.cpu().numpy()
    S = librosa.feature.melspectrogram(
        y=wave_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=2.0
    )
    log_mel = librosa.power_to_db(S, ref=np.max)
    return torch.from_numpy(log_mel).to(torch.float32)


def normalize_wave_rms(
    wave: torch.Tensor,
    target_rms: float = 0.1,
) -> torch.Tensor:
    """RMS normalise waveform to target RMS amplitude.

    wave: 1-D tensor.
    """
    if wave.dim() > 1:
        wave = wave.squeeze(0)
    rms = torch.sqrt(torch.mean(wave.pow(2))).clamp(min=1e-9)
    return wave * (target_rms / rms)
