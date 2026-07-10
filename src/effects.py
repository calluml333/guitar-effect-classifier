"""DSP effect implementations for dataset generation.

Functions accept and return 1-D NumPy arrays (float32) and operate at a given sample rate.
"""
from typing import Dict, Optional

import numpy as np
from scipy.signal import lfilter

from src.utils import ensure_mono


def apply_overdrive(wave: np.ndarray, gain: float = 2.0, tone: float = 0.7) -> np.ndarray:
    """Simple overdrive: gain stage + soft clipping + simple tone control.

    tone in [0,1] mixes between bright (1.0) and dark (0.0).
    """
    x = ensure_mono(wave).astype(np.float32)
    x = x * gain
    # soft clip
    x = np.tanh(x)
    # simple tone: single-pole IIR lowpass, y[n] = a*x[n] + (1-a)*y[n-1]
    a = 0.01 + 0.99 * tone
    y = lfilter([a], [1.0, -(1.0 - a)], x).astype(np.float32)
    # normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def apply_distortion(wave: np.ndarray, drive: float = 5.0, threshold: float = 0.6) -> np.ndarray:
    """Harder distortion: high gain then hard clip with optional smoothing.
    threshold in (0,1].
    """
    x = ensure_mono(wave).astype(np.float32)
    x = x * drive
    x = np.clip(x, -threshold, threshold)
    # normalize
    x = x / (threshold + 1e-9)
    return x


def apply_fuzz(wave: np.ndarray, gain: float = 10.0, bias: float = 0.0) -> np.ndarray:
    """Fuzz implemented with heavy waveshaping and optional DC bias.
    Produces square-ish clipping and high harmonic content.
    """
    x = ensure_mono(wave).astype(np.float32)
    x = x * gain + bias
    # aggressive non-linearity: sign(x) * (1 - exp(-abs(x)))
    y = np.sign(x) * (1.0 - np.exp(-np.abs(x)))
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def apply_chorus(wave: np.ndarray, sr: int, depth_ms: float = 10.0, rate_hz: float = 0.8, mix: float = 0.5) -> np.ndarray:
    """Basic chorus: LFO modulates a short fractional delay and mixes with dry signal.
    depth_ms: maximum modulation in milliseconds
    rate_hz: LFO frequency
    """
    x = ensure_mono(wave).astype(np.float32)
    n = x.shape[0]
    t = np.arange(n) / sr
    depth = depth_ms / 1000.0
    lfo = (np.sin(2 * np.pi * rate_hz * t) + 1.0) / 2.0  # 0..1
    delay_samples = lfo * depth * sr  # continuous per-sample delay, always >= 0

    floor_delay = np.floor(delay_samples).astype(np.int64)
    frac = (delay_samples - floor_delay).astype(np.float32)
    idx = np.arange(n) - floor_delay  # always < n since floor_delay >= 0

    idx0 = np.clip(idx, 0, n - 1)
    idx1 = np.clip(idx - 1, 0, n - 1)
    interpolated = (1 - frac) * x[idx0] + frac * x[idx1]
    delayed = np.where(idx >= 1, interpolated, np.where(idx == 0, x[idx0], 0.0))

    out = (1.0 - mix) * x + mix * delayed
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out.astype(np.float32)


def apply_delay(wave: np.ndarray, sr: int, delay_ms: float = 400.0, feedback: float = 0.35, mix: float = 0.5) -> np.ndarray:
    """Simple single-tap delay with feedback and lowpass on repeats.
    """
    x = ensure_mono(wave).astype(np.float32)
    delay_samps = int(sr * (delay_ms / 1000.0))
    out = np.zeros(x.shape[0] + delay_samps * 4, dtype=np.float32)
    out[: x.shape[0]] = x
    for i in range(x.shape[0]):
        out[i + delay_samps] += x[i] * mix
    # feedback loop
    for i in range(x.shape[0] + delay_samps):
        if i + delay_samps < out.shape[0]:
            out[i + delay_samps] += out[i] * feedback
    out = out[: x.shape[0]]
    # normalize
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out


def synthetic_ir(length_s: float, sr: int, decay: float = 2.0) -> np.ndarray:
    """Generate a simple exponential-decay impulse response (mono).
    decay: larger values -> faster decay.
    """
    n = int(length_s * sr)
    t = np.linspace(0, length_s, n)
    ir = np.random.randn(n) * np.exp(-decay * t)
    ir = ir / (np.max(np.abs(ir)) + 1e-9)
    return ir.astype(np.float32)


def apply_reverb(wave: np.ndarray, sr: int, ir: Optional[np.ndarray] = None, ir_len_s: float = 1.0, decay: float = 3.0, mix: float = 0.5) -> np.ndarray:
    """Apply reverb by convolving with an IR. If no IR provided, make a synthetic one.
    """
    x = ensure_mono(wave).astype(np.float32)
    if ir is None:
        ir = synthetic_ir(ir_len_s, sr, decay=decay)
    # convolution (numpy)
    y = np.convolve(x, ir)[: x.shape[0]]
    out = (1 - mix) * x + mix * y
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out


def apply_effect_by_name(wave: np.ndarray, sr: int, name: str, params: Dict) -> np.ndarray:
    name = name.lower()
    if name == "clean":
        return ensure_mono(wave).astype(np.float32)
    if name == "overdrive":
        return apply_overdrive(wave, gain=params.get("gain", 2.0), tone=params.get("tone", 0.7))
    if name == "distortion":
        return apply_distortion(wave, drive=params.get("drive", 5.0), threshold=params.get("threshold", 0.6))
    if name == "fuzz":
        return apply_fuzz(wave, gain=params.get("gain", 10.0), bias=params.get("bias", 0.0))
    if name == "chorus":
        return apply_chorus(wave, sr=sr, depth_ms=params.get("depth_ms", 10.0), rate_hz=params.get("rate_hz", 0.8), mix=params.get("mix", 0.5))
    if name == "delay":
        return apply_delay(wave, sr=sr, delay_ms=params.get("delay_ms", 400.0), feedback=params.get("feedback", 0.35), mix=params.get("mix", 0.5))
    if name == "reverb":
        return apply_reverb(wave, sr=sr, ir=params.get("ir", None), ir_len_s=params.get("ir_len_s", 1.0), decay=params.get("decay", 3.0), mix=params.get("mix", 0.5))
    raise ValueError(f"Unknown effect: {name}")
