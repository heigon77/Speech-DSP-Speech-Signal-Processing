"""
filters.py
----------
Digital filters for speech signal processing.

Filter types implemented:
  - Butterworth (IIR): lowpass, highpass, bandpass, bandstop
  - FIR (Hamming window)
  - Pre-emphasis filter (standard ASR pre-processing)
  - Moving average smoother
"""

import numpy as np
from scipy.signal import butter, firwin, lfilter, filtfilt, freqz


def _nyquist_check(freq, fs):
    """Ensure the given frequency is below the Nyquist limit."""
    nyq = fs / 2
    if np.any(np.array(freq) >= nyq):
        raise ValueError(f"Frequency {freq} Hz >= Nyquist ({nyq} Hz)")


# ─────────────────────────── IIR FILTERS (Butterworth) ─────────────────────────

def lowpass_filter(signal: np.ndarray, cutoff: float, fs: int = 16000,
                   order: int = 5) -> np.ndarray:
    """
    Butterworth lowpass filter.
    Attenuates frequencies above `cutoff`.
    Uses filtfilt (forward-backward) for zero-phase response.
    """
    _nyquist_check(cutoff, fs)
    b, a = butter(order, cutoff / (fs / 2), btype='low')
    return filtfilt(b, a, signal)


def highpass_filter(signal: np.ndarray, cutoff: float, fs: int = 16000,
                    order: int = 5) -> np.ndarray:
    """Butterworth highpass filter. Attenuates frequencies below `cutoff`."""
    _nyquist_check(cutoff, fs)
    b, a = butter(order, cutoff / (fs / 2), btype='high')
    return filtfilt(b, a, signal)


def bandpass_filter(signal: np.ndarray, f_low: float, f_high: float,
                    fs: int = 16000, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter between f_low and f_high."""
    _nyquist_check([f_low, f_high], fs)
    b, a = butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)


def bandstop_filter(signal: np.ndarray, f_low: float, f_high: float,
                    fs: int = 16000, order: int = 4) -> np.ndarray:
    """
    Butterworth bandstop (notch) filter.
    Useful for removing power-line interference (50/60 Hz).
    """
    _nyquist_check([f_low, f_high], fs)
    b, a = butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype='bandstop')
    return filtfilt(b, a, signal)


# ─────────────────────────── FIR FILTER ────────────────────────────────────────

def fir_filter(signal: np.ndarray, cutoff: float, fs: int = 16000,
               num_taps: int = 101, filter_type: str = 'lowpass') -> np.ndarray:
    """
    Hamming-windowed FIR filter.

    Advantages over IIR:
     - Linear phase (no phase distortion)
     - Inherently stable
     - Ideal for offline speech processing

    Args:
        num_taps:    Number of coefficients (should be odd for symmetry)
        filter_type: 'lowpass', 'highpass', 'bandpass', or 'bandstop'
    """
    nyq = fs / 2
    if isinstance(cutoff, (list, tuple)):
        norm_cutoff = [f / nyq for f in cutoff]
    else:
        norm_cutoff = cutoff / nyq

    taps = firwin(num_taps, norm_cutoff,
                  pass_zero=(filter_type in ['lowpass', 'bandstop']),
                  window='hamming')
    return np.convolve(signal, taps, mode='same')


def frequency_response(b, a, fs: int = 16000, n_points: int = 512):
    """
    Compute the frequency response of an IIR filter.
    Returns (freqs_hz, magnitude_db, phase_deg).
    """
    w, h = freqz(b, a, worN=n_points)
    freqs_hz     = w * fs / (2 * np.pi)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
    phase_deg    = np.angle(h, deg=True)
    return freqs_hz, magnitude_db, phase_deg


# ─────────────────────────── SPEECH PRE-PROCESSING ─────────────────────────────

def pre_emphasis(signal: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Pre-emphasis filter: y[n] = x[n] - coef * x[n-1]

    Purpose:
     - Boosts high frequencies (compensates the -6 dB/octave vocal tract roll-off)
     - Improves SNR at high frequencies
     - Standard step before MFCC extraction in ASR pipelines

    Typical coefficient: 0.95 to 0.97
    """
    return np.append(signal[0], signal[1:] - coef * signal[:-1])


def moving_average(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Moving average smoother (equal-coefficient FIR filter).
    Equivalent to a simple lowpass filter.
    """
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')
