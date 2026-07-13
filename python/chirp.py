"""LFM chirp synthesis, matched filtering, and synthetic echo generation."""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp, correlate, fftconvolve

from config import (
    CHIRP_MODES,
    CHIRP_PARAMS,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_M_S,
)


def generate_lfm(
    mode: str = "GEOMETRY",
    fs: float = SAMPLE_RATE_HZ,
    amplitude: float = 0.8,
) -> np.ndarray:
    """
    Generate a linear frequency-modulated (LFM) chirp for the given mode.

    Args:
        mode: One of GEOMETRY, MATERIAL, GLASS_PROBE
        fs: Sample rate in Hz

    Returns:
        1D float64 chirp waveform
    """
    if mode not in CHIRP_PARAMS:
        raise ValueError(f"Unknown chirp mode: {mode}. Choose from {CHIRP_MODES}")

    params = CHIRP_PARAMS[mode]
    duration_s = params["duration_ms"] / 1000.0
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    waveform = chirp(
        t,
        f0=params["f0"],
        f1=params["f1"],
        t1=duration_s,
        method="linear",
    )
    # Apply Hann window to reduce sidelobes
    window = np.hanning(n_samples)
    return (waveform * window * amplitude).astype(np.float64)


def matched_filter(
    recorded: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    Cross-correlate recorded signal with transmitted chirp template.

    Args:
        recorded: 1D recorded echo signal
        template: 1D transmitted chirp

    Returns:
        Correlation envelope (magnitude of complex correlation)
    """
    corr = correlate(recorded, template, mode="full", method="fft")
    return np.abs(corr)


def time_of_arrival(
    correlation: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    min_range_m: float = 0.1,
    max_range_m: float = 5.0,
) -> tuple[float, int]:
    """
    Find time-of-arrival from correlation peak.

    Returns:
        (range_m, peak_index)
    """
    min_delay = int(2 * min_range_m / SPEED_OF_SOUND_M_S * fs)
    max_delay = int(2 * max_range_m / SPEED_OF_SOUND_M_S * fs)
    max_delay = min(max_delay, len(correlation) - 1)

    if min_delay >= max_delay:
        return 0.0, 0

    search_region = correlation[min_delay:max_delay]
    peak_idx = int(np.argmax(search_region)) + min_delay
    toa_s = peak_idx / fs
    range_m = toa_s * SPEED_OF_SOUND_M_S / 2.0
    return range_m, peak_idx


def synthetic_echo(
    mode: str = "GEOMETRY",
    range_m: float = 2.0,
    material_idx: int = 0,
    bearing_deg: float = 0.0,
    fs: float = SAMPLE_RATE_HZ,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate fake echo response for pipeline testing without hardware.

    Different materials get different spectral attenuation profiles.
    """
    rng = np.random.default_rng(seed)
    chirp_wav = generate_lfm(mode, fs=fs)

    # Total recording length: chirp + max delay + tail
    record_len = int((range_m * 2 / SPEED_OF_SOUND_M_S + 0.05) * fs) + len(chirp_wav)
    record_len = max(record_len, len(chirp_wav) * 3)
    recorded = np.zeros(record_len, dtype=np.float64)

    # Material-dependent attenuation (higher idx = more absorption at high freq)
    attenuation = 0.3 + 0.1 * material_idx
    delay_samples = int(2 * range_m / SPEED_OF_SOUND_M_S * fs)

    # Place attenuated, slightly noisy echo at delay
    echo = chirp_wav * attenuation
    echo += 0.02 * rng.standard_normal(len(echo))

    end = min(delay_samples + len(echo), record_len)
    recorded[delay_samples:end] += echo[: end - delay_samples]

    # Add diffuse reverberation tail for non-glass materials
    if material_idx != 2:  # not glass
        tail_len = int(0.08 * fs)
        tail_start = delay_samples + len(chirp_wav)
        if tail_start + tail_len < record_len:
            tail = 0.05 * attenuation * rng.standard_normal(tail_len)
            tail *= np.exp(-np.arange(tail_len) / (tail_len * 0.3))
            recorded[tail_start : tail_start + tail_len] += tail

    # Background noise
    recorded += 0.005 * rng.standard_normal(record_len)

    return recorded


def synthetic_multichannel_echo(
    mode: str = "GEOMETRY",
    range_m: float = 2.0,
    material_idx: int = 0,
    bearing_deg: float = 0.0,
    fs: float = SAMPLE_RATE_HZ,
    n_mics: int = 4,
    mic_spacing_m: float = 0.03,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate multichannel echo with inter-mic phase delay for DOA.

    Returns:
        (n_samples, n_mics) array
    """
    base = synthetic_echo(mode, range_m, material_idx, bearing_deg, fs, seed)
    channels = np.zeros((len(base), n_mics), dtype=np.float64)

    bearing_rad = np.radians(bearing_deg)
    for mic in range(n_mics):
        # ITD: time difference based on mic position and bearing
        mic_offset = (mic - (n_mics - 1) / 2) * mic_spacing_m
        itd_s = mic_offset * np.sin(bearing_rad) / SPEED_OF_SOUND_M_S
        delay = int(itd_s * fs)
        if delay >= 0:
            channels[delay:, mic] = base[: len(base) - delay]
        else:
            channels[: delay, mic] = base[-delay:]

    return channels
