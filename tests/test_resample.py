"""Tests for resample.resample_to_target."""
import os
import sys

import numpy as np

try:
    import pytest
except ImportError:  # standalone runner fallback
    class _PytestShim:  # noqa: D401
        @staticmethod
        def raises(exc):
            class _Ctx:
                def __enter__(self_): return self_
                def __exit__(self_, et, ev, tb): return et is not None and issubclass(et, exc)
            return _Ctx()
    pytest = _PytestShim()  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resample import TARGET_FS, TARGET_SAMPLES, resample_to_target


def _sine(fs: float, duration: float, freq: float, n_leads: int = 12) -> np.ndarray:
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.tile(sig[:, None], (1, n_leads))


def _gaussian_pulse(fs: float, duration: float, peak_t: float,
                    sigma_s: float = 0.01, n_leads: int = 12) -> np.ndarray:
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    pulse = np.exp(-((t - peak_t) ** 2) / (2 * sigma_s ** 2)).astype(np.float32)
    return np.tile(pulse[:, None], (1, n_leads))


def test_identity_500hz_10s():
    sig = _sine(500, 10.0, freq=5.0)
    out, fs, msg = resample_to_target(sig, 500.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert fs == TARGET_FS
    assert msg == ""
    assert np.allclose(out, sig.astype(np.float32))


def test_downsample_1000hz_to_500hz():
    sig = _sine(1000, 10.0, freq=5.0)
    out, fs, msg = resample_to_target(sig, 1000.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert fs == TARGET_FS
    assert "1000" in msg and "500" in msg


def test_upsample_250hz_to_500hz():
    sig = _sine(250, 10.0, freq=5.0)
    out, fs, msg = resample_to_target(sig, 250.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert fs == TARGET_FS
    assert "250" in msg


def test_rational_360hz_to_500hz():
    sig = _sine(360, 10.0, freq=5.0)
    out, fs, msg = resample_to_target(sig, 360.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert fs == TARGET_FS


def test_trim_500hz_15s():
    sig = _sine(500, 15.0, freq=5.0)
    assert sig.shape[0] == 7500
    out, fs, msg = resample_to_target(sig, 500.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert "trim" in msg
    assert np.allclose(out, sig[:TARGET_SAMPLES])


def test_pad_500hz_5s():
    sig = _sine(500, 5.0, freq=5.0)
    assert sig.shape[0] == 2500
    out, fs, msg = resample_to_target(sig, 500.0)
    assert out.shape == (TARGET_SAMPLES, 12)
    assert "pad" in msg
    assert np.allclose(out[:2500], sig)
    edge_value = sig[-1]
    assert np.allclose(out[2500:], edge_value)


def test_sine_frequency_preserved_after_downsample():
    sig = _sine(1000, 10.0, freq=10.0, n_leads=1)
    out, _, _ = resample_to_target(sig, 1000.0)
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(out.shape[0], d=1.0 / TARGET_FS)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 10.0) < 0.2


def test_sine_frequency_preserved_after_upsample():
    sig = _sine(250, 10.0, freq=10.0, n_leads=1)
    out, _, _ = resample_to_target(sig, 250.0)
    spectrum = np.abs(np.fft.rfft(out[:, 0]))
    freqs = np.fft.rfftfreq(out.shape[0], d=1.0 / TARGET_FS)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 10.0) < 0.2


def test_r_peak_amplitude_preserved_downsample():
    sig = _gaussian_pulse(1000, 10.0, peak_t=5.0, n_leads=1)
    out, _, _ = resample_to_target(sig, 1000.0)
    assert abs(out.max() - 1.0) < 0.05


def test_r_peak_amplitude_preserved_upsample():
    sig = _gaussian_pulse(250, 10.0, peak_t=5.0, n_leads=1)
    out, _, _ = resample_to_target(sig, 250.0)
    assert abs(out.max() - 1.0) < 0.05


def test_dtype_is_float32():
    sig = _sine(500, 10.0, freq=5.0).astype(np.float64)
    out, _, _ = resample_to_target(sig, 500.0)
    assert out.dtype == np.float32


def test_rejects_1d():
    with pytest.raises(ValueError):
        resample_to_target(np.zeros(5000, dtype=np.float32), 500.0)


def test_rejects_nonpositive_fs():
    with pytest.raises(ValueError):
        resample_to_target(np.zeros((5000, 12), dtype=np.float32), 0)
