"""Tests for the DSP effect implementations in src/effects.py."""
import numpy as np
import pytest

from src.effects import (
    apply_chorus,
    apply_delay,
    apply_distortion,
    apply_effect_by_name,
    apply_fuzz,
    apply_overdrive,
    apply_reverb,
    synthetic_ir,
)

SR = 16000


@pytest.fixture
def sine_wave():
    t = np.arange(SR) / SR
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def test_apply_overdrive_shape_dtype_and_range(sine_wave):
    y = apply_overdrive(sine_wave, gain=3.0, tone=0.5)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_distortion_shape_dtype_and_range(sine_wave):
    y = apply_distortion(sine_wave, drive=8.0, threshold=0.5)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_fuzz_shape_dtype_and_range(sine_wave):
    y = apply_fuzz(sine_wave, gain=12.0, bias=0.1)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_chorus_shape_dtype_and_range(sine_wave):
    y = apply_chorus(sine_wave, sr=SR, depth_ms=10.0, rate_hz=1.0, mix=0.5)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_delay_shape_dtype_and_range(sine_wave):
    y = apply_delay(sine_wave, sr=SR, delay_ms=200.0, feedback=0.4, mix=0.5)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_reverb_shape_dtype_and_range(sine_wave):
    y = apply_reverb(sine_wave, sr=SR, ir_len_s=0.2, decay=3.0, mix=0.5)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32
    assert np.abs(y).max() <= 1.0 + 1e-6


def test_apply_reverb_accepts_explicit_ir(sine_wave):
    ir = synthetic_ir(0.1, SR, decay=2.0)
    y = apply_reverb(sine_wave, sr=SR, ir=ir, mix=0.6)
    assert y.shape == sine_wave.shape


def test_synthetic_ir_shape_and_dtype():
    ir = synthetic_ir(0.5, SR, decay=2.0)
    assert ir.shape == (int(0.5 * SR),)
    assert ir.dtype == np.float32
    assert np.abs(ir).max() <= 1.0 + 1e-6


def test_ensure_mono_downmixes_stereo_input():
    stereo = np.random.randn(500, 2).astype(np.float32)
    y = apply_overdrive(stereo)
    assert y.shape == (500,)


@pytest.mark.parametrize(
    "effect,params",
    [
        ("clean", {}),
        ("overdrive", {"gain": 2.0, "tone": 0.5}),
        ("distortion", {"drive": 5.0, "threshold": 0.6}),
        ("fuzz", {"gain": 10.0, "bias": 0.0}),
        ("chorus", {"depth_ms": 10.0, "rate_hz": 0.8, "mix": 0.5}),
        ("delay", {"delay_ms": 300.0, "feedback": 0.3, "mix": 0.4}),
        ("reverb", {"ir_len_s": 0.3, "decay": 2.0, "mix": 0.5}),
    ],
)
def test_apply_effect_by_name_dispatches_all_known_effects(sine_wave, effect, params):
    y = apply_effect_by_name(sine_wave, SR, effect, params)
    assert y.shape == sine_wave.shape
    assert y.dtype == np.float32


def test_apply_effect_by_name_is_case_insensitive(sine_wave):
    y = apply_effect_by_name(sine_wave, SR, "OVERDRIVE", {})
    assert y.shape == sine_wave.shape


def test_apply_effect_by_name_rejects_unknown_effect(sine_wave):
    with pytest.raises(ValueError, match="Unknown effect"):
        apply_effect_by_name(sine_wave, SR, "flanger", {})
