"""
vad.py
------
Voice Activity Detection (VAD).

Two methods implemented:
  1. Energy + ZCR VAD (classic, no ML)
  2. Adaptive energy threshold VAD
"""

import numpy as np
from .features import rms_energy, zero_crossing_rate


def energy_zcr_vad(signal: np.ndarray, fs: int = 16000,
                   energy_threshold: float = None,
                   zcr_threshold: float = 0.15) -> tuple:
    """
    Classic VAD combining RMS energy and ZCR.

    Logic:
     - Frame is VOICED if: energy > threshold_E  AND  zcr < threshold_ZCR
       (voiced sounds have high energy and low ZCR)
     - Frame is SILENCE otherwise

    If energy_threshold is None, it is estimated automatically as 3x the
    mean energy of the first 200 ms (assumed silence).

    Returns:
        rms:  per-frame RMS energy
        zcr:  per-frame ZCR
        mask: boolean array (True = voiced)
    """
    frame_size = int(fs * 0.025)
    hop        = int(fs * 0.010)

    rms = rms_energy(signal, frame_size, hop)
    zcr = zero_crossing_rate(signal, frame_size, hop)

    if energy_threshold is None:
        # Auto-estimate: assume first 200 ms is silence
        n_silence       = max(1, int(0.200 / 0.010))
        energy_threshold = 3 * np.mean(rms[:n_silence]) + 1e-9

    energy_mask = rms > energy_threshold
    zcr_mask    = zcr < zcr_threshold
    mask        = energy_mask & zcr_mask

    # Smoothing (5-frame window) to avoid fragmentation
    kernel      = np.ones(5) / 5
    smooth_mask = np.convolve(mask.astype(float), kernel, mode='same') > 0.4

    return rms, zcr, smooth_mask


def adaptive_vad(signal: np.ndarray, fs: int = 16000,
                 alpha: float = 0.99) -> tuple:
    """
    VAD with an adaptive energy threshold.

    Continuously updates the silence model using an exponential moving
    average (similar to ITU-T G.729 Annex B).

    Args:
        alpha: forgetting factor (0.99 = slow adaptation)

    Returns:
        rms:       per-frame RMS energy
        mask:      boolean array (True = voiced)
        threshold: adaptive threshold evolution
    """
    frame_size = int(fs * 0.025)
    hop        = int(fs * 0.010)
    rms        = rms_energy(signal, frame_size, hop)

    n_frames  = len(rms)
    mask      = np.zeros(n_frames, dtype=bool)
    threshold = np.zeros(n_frames)

    # Initialize with first 10 frames (assumed silence)
    E_min = np.mean(rms[:10]) + 1e-9

    for i in range(n_frames):
        thr         = E_min * 3
        threshold[i] = thr
        mask[i]     = rms[i] > thr

        # Update silence model only on silent frames
        if not mask[i]:
            E_min = alpha * E_min + (1 - alpha) * rms[i]

    return rms, mask, threshold


def extract_voiced_segments(signal: np.ndarray, mask: np.ndarray,
                            fs: int = 16000, hop_ms: float = 10.0) -> list:
    """
    Extract voiced regions from the signal based on a VAD mask.

    Returns a list of tuples (start_s, end_s, segment_array).
    """
    hop      = int(fs * hop_ms / 1000)
    segments = []
    in_voice = False
    start_frame = 0

    for i, voiced in enumerate(mask):
        if voiced and not in_voice:
            in_voice    = True
            start_frame = i
        elif not voiced and in_voice:
            in_voice    = False
            start_s     = start_frame * hop / fs
            end_s       = i * hop / fs
            start_samp  = start_frame * hop
            end_samp    = min(i * hop, len(signal))
            segments.append((start_s, end_s, signal[start_samp:end_samp]))

    # Close any open segment at end of signal
    if in_voice:
        start_s    = start_frame * hop / fs
        end_s      = len(signal) / fs
        start_samp = start_frame * hop
        segments.append((start_s, end_s, signal[start_samp:]))

    return segments
