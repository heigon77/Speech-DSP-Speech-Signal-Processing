"""
features.py
-----------
Feature extraction for speech signals.

Techniques implemented:
  1. STFT  - Short-Time Fourier Transform
  2. Power Spectrogram
  3. Mel Filter Bank
  4. MFCC  - Mel-Frequency Cepstral Coefficients
  5. Delta and Delta-Delta (dynamic coefficients)
  6. ZCR   - Zero Crossing Rate
  7. RMS Energy
  8. Fundamental frequency via autocorrelation (F0 / Pitch)
  9. Power Spectral Density via Welch's method
"""

import numpy as np
from scipy.signal import get_window, welch
from scipy.fft import fft, fftfreq


# ─────────────────────────── WINDOWING ─────────────────────────────────────────

def frame_signal(signal: np.ndarray, frame_size: int, hop: int,
                 window_type: str = 'hann') -> np.ndarray:
    """
    Split the signal into overlapping frames with windowing applied.

    Supported windows: 'hann', 'hamming', 'blackman', 'rectangular'

    Why windowing? The DFT assumes periodic input — discontinuities at
    frame edges cause spectral leakage. Windows taper the edges smoothly.

    - Hann:     good side-lobe suppression, general purpose
    - Hamming:  first null closer to main lobe, slightly narrower
    - Blackman: very low side lobes, lower frequency resolution
    """
    window = get_window(window_type, frame_size)
    n_frames = 1 + (len(signal) - frame_size) // hop
    frames = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        start = i * hop
        frames[i] = signal[start: start + frame_size] * window
    return frames


# ─────────────────────────── STFT ──────────────────────────────────────────────

def stft(signal: np.ndarray, fs: int = 16000,
         window_ms: float = 25.0,
         hop_ms: float = 10.0,
         window_type: str = 'hann',
         n_fft: int = None):
    """
    Short-Time Fourier Transform (STFT).

    Standard speech parameters:
      - Window: 20–30 ms (captures local stationarity of speech)
      - Hop:    10 ms (50–60% overlap)

    Returns:
        freqs     : frequency bins (Hz)
        times     : frame center times (s)
        magnitude : |STFT|  (n_frames x n_bins)
        phase     : angle of STFT
    """
    frame_size = int(fs * window_ms / 1000)
    hop = int(fs * hop_ms / 1000)
    if n_fft is None:
        n_fft = max(frame_size, 512)
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))   # power of 2 for efficient FFT

    frames = frame_signal(signal, frame_size, hop, window_type)

    # Zero-pad to n_fft
    if frames.shape[1] < n_fft:
        frames = np.pad(frames, ((0, 0), (0, n_fft - frames.shape[1])))

    spectrum = fft(frames, n=n_fft, axis=1)
    half = n_fft // 2 + 1
    spectrum = spectrum[:, :half]

    freqs   = fftfreq(n_fft, 1 / fs)[:half]
    n_frames = frames.shape[0]
    times    = np.arange(n_frames) * hop / fs + (frame_size / 2) / fs

    magnitude = np.abs(spectrum)
    phase     = np.angle(spectrum)
    return freqs, times, magnitude, phase


def power_spectrogram(signal: np.ndarray, fs: int = 16000, **kwargs) -> tuple:
    """
    Power spectrogram in dB.
    P(f,t) = 20 * log10(|STFT(f,t)| + eps)
    """
    freqs, times, magnitude, _ = stft(signal, fs, **kwargs)
    power_db = 20 * np.log10(magnitude + 1e-9)
    return freqs, times, power_db


# ─────────────────────────── MEL SCALE ─────────────────────────────────────────

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to Mel (O'Shaughnessy formula)."""
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel to Hz."""
    return 700 * (10 ** (mel / 2595) - 1)


def mel_filterbank(n_filters: int = 26, n_fft: int = 512,
                   fs: int = 16000, f_min: float = 0.0,
                   f_max: float = None) -> np.ndarray:
    """
    Build a triangular Mel filterbank.

    The Mel scale approximates human pitch perception:
     - Fine resolution at low frequencies
     - Coarse resolution at high frequencies

    Returns:
        filterbank: matrix (n_filters x n_fft//2+1)
    """
    if f_max is None:
        f_max = fs / 2

    mel_min = hz_to_mel(np.array([f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]

    # n_filters + 2 equally spaced points in Mel
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points  = mel_to_hz(mel_points)

    # Map to FFT bins
    bins   = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    n_bins = n_fft // 2 + 1
    fb     = np.zeros((n_filters, n_bins))

    for m in range(1, n_filters + 1):
        f_start  = bins[m - 1]
        f_center = bins[m]
        f_end    = bins[m + 1]

        for k in range(f_start, f_center):
            if f_center != f_start:
                fb[m - 1, k] = (k - f_start) / (f_center - f_start)
        for k in range(f_center, f_end):
            if f_end != f_center:
                fb[m - 1, k] = (f_end - k) / (f_end - f_center)

    return fb


# ─────────────────────────── MFCC ──────────────────────────────────────────────

def mfcc(signal: np.ndarray, fs: int = 16000,
         n_mfcc: int = 13, n_filters: int = 26,
         window_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """
    Mel-Frequency Cepstral Coefficients (MFCC).

    Full pipeline:
      1. Pre-emphasis
      2. STFT + power spectrum
      3. Mel filterbank
      4. log() — dynamic range compression (perceptual)
      5. DCT  — decorrelation, Mel cepstrum

    MFCCs are the most widely used features in ASR.
    Returns: (n_frames x n_mfcc)
    """
    from .filters import pre_emphasis
    from scipy.fft import dct

    signal_pe  = pre_emphasis(signal)
    frame_size = int(fs * window_ms / 1000)
    hop        = int(fs * hop_ms / 1000)
    n_fft      = int(2 ** np.ceil(np.log2(frame_size)))
    n_fft      = max(n_fft, 512)

    frames = frame_signal(signal_pe, frame_size, hop, 'hamming')
    if frames.shape[1] < n_fft:
        frames = np.pad(frames, ((0, 0), (0, n_fft - frames.shape[1])))

    # Power spectrum
    spectrum = fft(frames, n=n_fft, axis=1)
    n_bins   = n_fft // 2 + 1
    power    = (np.abs(spectrum[:, :n_bins]) ** 2) / n_fft

    # Mel filterbank
    fb        = mel_filterbank(n_filters, n_fft, fs)
    mel_energy = np.dot(power, fb.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel    = np.log(mel_energy)

    # DCT-II for decorrelation
    coeffs = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return coeffs


def delta(coefficients: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Delta (velocity) coefficients of MFCCs.
    Δc[t] = Σ_{n=1}^{N} n*(c[t+n] - c[t-n]) / (2*Σn²)

    Delta-delta (acceleration) = delta(delta(coeffs)).
    Together, MFCCs + Δ + ΔΔ = 39 features (standard in HMM-GMM ASR).
    """
    n_frames, n_coeffs = coefficients.shape
    deltas = np.zeros_like(coefficients)
    denom  = 2 * sum(n ** 2 for n in range(1, N + 1))
    for t in range(n_frames):
        for n in range(1, N + 1):
            t_prev = max(0, t - n)
            t_next = min(n_frames - 1, t + n)
            deltas[t] += n * (coefficients[t_next] - coefficients[t_prev])
    return deltas / denom


# ─────────────────────────── TEMPORAL FEATURES ─────────────────────────────────

def zero_crossing_rate(signal: np.ndarray, frame_size: int = 400,
                       hop: int = 160) -> np.ndarray:
    """
    Zero Crossing Rate (ZCR).

    Discriminates voiced sounds (vowels, low ZCR) from unvoiced sounds
    (fricatives like /s/, high ZCR). Foundation for simple VAD systems.
    """
    n_frames = 1 + (len(signal) - frame_size) // hop
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start: start + frame_size]
        zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * frame_size)
    return zcr


def rms_energy(signal: np.ndarray, frame_size: int = 400,
               hop: int = 160) -> np.ndarray:
    """
    Per-frame RMS energy.
    E[n] = sqrt(1/N * Σ x²)

    Used in VAD and volume normalization.
    """
    n_frames = 1 + (len(signal) - frame_size) // hop
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start: start + frame_size]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


# ─────────────────────────── PITCH (F0) ────────────────────────────────────────

def detect_pitch(signal: np.ndarray, fs: int = 16000,
                 f_min: float = 60.0, f_max: float = 400.0) -> tuple:
    """
    Fundamental frequency (F0 / Pitch) detection via autocorrelation.

    Method:
     - Compute normalized autocorrelation for each frame
     - Find the first peak within the human pitch range (60–400 Hz)
     - F0 male voice: 85–180 Hz; female: 165–255 Hz
     - Returns 0 for unvoiced frames

    Returns:
        times : frame center times (s)
        f0    : estimated fundamental frequency (Hz), 0 = unvoiced
    """
    frame_size = int(fs * 0.025)    # 25 ms
    hop        = int(fs * 0.010)    # 10 ms
    lag_min    = int(fs / f_max)
    lag_max    = int(fs / f_min)

    n_frames = 1 + (len(signal) - frame_size) // hop
    f0    = np.zeros(n_frames)
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = signal[start: start + frame_size].copy()
        frame -= np.mean(frame)   # DC removal

        # Normalized autocorrelation
        autocorr  = np.correlate(frame, frame, mode='full')
        autocorr  = autocorr[len(autocorr) // 2:]
        autocorr /= (autocorr[0] + 1e-9)

        # Peak search within pitch range
        segment = autocorr[lag_min: lag_max + 1]
        if len(segment) > 0 and np.max(segment) > 0.3:
            lag    = np.argmax(segment) + lag_min
            f0[i]  = fs / lag
        else:
            f0[i] = 0.0

        times[i] = (start + frame_size / 2) / fs

    return times, f0


# ─────────────────────────── WELCH PSD ─────────────────────────────────────────

def welch_psd(signal: np.ndarray, fs: int = 16000,
              nperseg: int = 1024) -> tuple:
    """
    Power Spectral Density estimate using Welch's method.

    Advantage over direct FFT: reduced variance through averaged,
    overlapping periodograms.
    Returns (freqs_hz, psd_db).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    psd_db = 10 * np.log10(psd + 1e-12)
    return freqs, psd_db
