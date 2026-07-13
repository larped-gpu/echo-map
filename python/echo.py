"""Echo processing: TOA range, DOA bearing, spectral features, glass heuristic."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis

from config import (
    GLASS_KURTOSIS_THRESHOLD,
    MFCC_COEFFS,
    MFCC_HOP,
    MFCC_NFFT,
    MIC_SPACING_M,
    NUM_MICS,
    SAMPLE_RATE_HZ,
    SPEED_OF_SOUND_M_S,
)
from chirp import matched_filter, time_of_arrival


def extract_mfcc_features(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    n_mfcc: int = MFCC_COEFFS,
) -> np.ndarray:
    """
    Extract MFCC-like spectral features from echo signal.

    Uses STFT magnitude as a lightweight stand-in when librosa is unavailable.
    Returns flattened feature vector.
    """
    try:
        import librosa

        mfccs = librosa.feature.mfcc(
            y=signal.astype(np.float32),
            sr=int(fs),
            n_mfcc=n_mfcc,
            n_fft=MFCC_NFFT,
            hop_length=MFCC_HOP,
        )
        return mfccs.flatten()
    except ImportError:
        # Fallback: log-magnitude STFT bins
        _, _, Zxx = stft(signal, fs=fs, nperseg=MFCC_NFFT, noverlap=MFCC_NFFT - MFCC_HOP)
        mag = np.abs(Zxx)
        log_mag = np.log1p(mag)
        # Take mean across time, pad/truncate to fixed size
        feat = log_mag.mean(axis=1)
        target_len = n_mfcc * 10
        if len(feat) < target_len:
            feat = np.pad(feat, (0, target_len - len(feat)))
        else:
            feat = feat[:target_len]
        return feat


def spectral_centroid(signal: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> float:
    """Compute spectral centroid in Hz."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
    if spectrum.sum() < 1e-10:
        return 0.0
    return float(np.sum(freqs * spectrum) / np.sum(spectrum))


def early_late_energy_ratio(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    early_ms: float = 5.0,
    late_ms: float = 50.0,
) -> float:
    """Ratio of early vs late echo energy (material discrimination cue)."""
    early_n = int(early_ms / 1000.0 * fs)
    late_n = int(late_ms / 1000.0 * fs)
    early_energy = np.sum(signal[:early_n] ** 2) + 1e-10
    late_energy = np.sum(signal[early_n:late_n] ** 2) + 1e-10
    return float(early_energy / late_energy)


def estimate_bearing(
    correlations: list[np.ndarray],
    fs: float = SAMPLE_RATE_HZ,
    mic_spacing_m: float = MIC_SPACING_M,
) -> float:
    """
    Estimate bearing (degrees) from inter-mic TOA differences.

    Args:
        correlations: list of correlation envelopes, one per mic

    Returns:
        Bearing in degrees (-90 to +90)
    """
    if len(correlations) < 2:
        return 0.0

    toas = []
    for corr in correlations:
        _, peak_idx = time_of_arrival(corr, fs=fs)
        toas.append(peak_idx / fs)

    # Use first and last mic for widest baseline
    itd = toas[-1] - toas[0]
    baseline = (len(correlations) - 1) * mic_spacing_m
    if baseline < 1e-6:
        return 0.0

    sin_theta = np.clip(itd * SPEED_OF_SOUND_M_S / baseline, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_theta)))


def glass_probe_heuristic(
    correlation: np.ndarray,
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
) -> tuple[bool, float]:
    """
    Detect glass/mirror signature: sharp early peak + low spectral spread.

    Returns:
        (is_glass, confidence)
    """
    peak_kurt = float(kurtosis(correlation))
    centroid = spectral_centroid(signal, fs)
    el_ratio = early_late_energy_ratio(signal, fs)

    is_sharp = peak_kurt > GLASS_KURTOSIS_THRESHOLD
    is_high_freq = centroid > 8000.0
    is_specular = el_ratio > 3.0

    score = sum([is_sharp, is_high_freq, is_specular]) / 3.0
    return score > 0.6, score


def process_echo(
    recorded: np.ndarray,
    template: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
) -> dict:
    """
    Full echo processing for a single channel.

    Returns dict with range_m, features, correlation, glass hints.
    """
    if recorded.ndim == 2:
        recorded = recorded[:, 0]

    corr = matched_filter(recorded, template)
    range_m, peak_idx = time_of_arrival(corr, fs=fs)

    # Extract features from echo tail
    echo_start = max(0, peak_idx - len(template) // 2)
    echo_end = min(len(recorded), peak_idx + len(template))
    echo_segment = recorded[echo_start:echo_end]

    features = extract_mfcc_features(echo_segment, fs=fs)
    is_glass, glass_conf = glass_probe_heuristic(corr, echo_segment, fs=fs)

    return {
        "range_m": range_m,
        "peak_idx": peak_idx,
        "features": features,
        "correlation": corr,
        "spectral_centroid": spectral_centroid(echo_segment, fs),
        "early_late_ratio": early_late_energy_ratio(echo_segment, fs),
        "is_glass": is_glass,
        "glass_confidence": glass_conf,
    }


def process_multichannel(
    recorded: np.ndarray,
    template: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
) -> dict:
    """
    Process all mic channels and fuse range + bearing.

    Args:
        recorded: (n_samples, n_mics)

    Returns:
        Fused result dict
    """
    if recorded.ndim == 1:
        recorded = recorded.reshape(-1, 1)

    n_mics = recorded.shape[1]
    correlations = []
    ranges = []
    all_features = []

    for mic in range(n_mics):
        result = process_echo(recorded[:, mic], template, fs=fs)
        correlations.append(result["correlation"])
        ranges.append(result["range_m"])
        all_features.append(result["features"])

    bearing_deg = estimate_bearing(correlations, fs=fs)
    median_range = float(np.median(ranges))

    # Fuse features by averaging across mics
    fused_features = np.mean(all_features, axis=0)

    # Glass check on strongest channel
    best_mic = int(np.argmax([np.max(c) for c in correlations]))
    best_result = process_echo(recorded[:, best_mic], template, fs=fs)

    return {
        "range_m": median_range,
        "bearing_deg": bearing_deg,
        "features": fused_features,
        "is_glass": best_result["is_glass"],
        "glass_confidence": best_result["glass_confidence"],
        "per_mic_ranges": ranges,
    }


def parse_odometry_line(line: str) -> tuple[int, int, int, float, float] | None:
    """Parse firmware CSV: timestamp_ms,left_ticks,right_ticks,heading_deg,servo_deg"""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) != 5:
        return None
    ts = int(parts[0])
    left = int(parts[1])
    right = int(parts[2])
    heading = float(parts[3])
    servo = float(parts[4])
    return ts, left, right, heading, servo
