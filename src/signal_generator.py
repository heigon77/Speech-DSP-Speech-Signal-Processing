"""
signal_generator.py
-------------------
Synthetic signal generation for speech signal processing demonstrations.
Includes: synthetic voice signals, noise, pure tones, and chirps.
"""

import numpy as np


def generate_pure_tone(freq: float, duration: float, fs: int = 16000, amplitude: float = 0.8) -> np.ndarray:
    """
    Generate a pure sinusoidal tone.

    Args:
        freq:      Frequency in Hz
        duration:  Duration in seconds
        fs:        Sampling rate (Hz)
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        NumPy array containing the signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def generate_synthetic_voice(duration: float = 2.0, fs: int = 16000) -> np.ndarray:
    """
    Simulate a synthetic voice signal using a simplified source-filter model.

    Source-filter model (Fant, 1960):
    - SOURCE: glottal pulse train (fundamental frequency F0 ~120 Hz)
    - FILTER: vocal tract formants (F1~700Hz, F2~1200Hz, F3~2500Hz)

    Returns:
        Normalized synthetic voice signal
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # --- SOURCE: glottal pulses (F0 = 120 Hz, male voice) ---
    F0 = 120.0
    pulses = np.zeros_like(t)
    period = int(fs / F0)
    for i in range(0, len(t), period):
        if i < len(pulses):
            # Asymmetric glottal pulse (Rosenberg model)
            mid = period // 3
            for k in range(min(period, len(pulses) - i)):
                if k < mid:
                    pulses[i + k] = np.sin(np.pi * k / mid) ** 2
                else:
                    pulses[i + k] = -0.5 * np.sin(np.pi * (k - mid) / (2 * (period - mid)))

    # --- FILTER: vocal tract resonances (formants) ---
    from scipy.signal import butter, lfilter

    def formant(signal, center_freq, bandwidth, sample_rate):
        """Bandpass filter mimicking a vocal tract formant."""
        w0 = center_freq / (sample_rate / 2)
        bw = bandwidth   / (sample_rate / 2)
        b, a = butter(2, [w0 - bw / 2, w0 + bw / 2], btype='band')
        return lfilter(b, a, signal)

    F1 = formant(pulses, 700,  150, fs)   # Vowel /a/
    F2 = formant(pulses, 1200, 200, fs)
    F3 = formant(pulses, 2500, 300, fs)

    voice = 0.6 * F1 + 0.3 * F2 + 0.1 * F3

    # Amplitude envelope (attack / decay)
    envelope = np.ones_like(voice)
    attack = int(0.05 * fs)
    decay  = int(0.10 * fs)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    voice *= envelope

    # Normalize
    voice = voice / (np.max(np.abs(voice)) + 1e-9)
    return voice


def generate_chirp(f_start: float, f_end: float, duration: float, fs: int = 16000) -> np.ndarray:
    """
    Generate a linear frequency sweep (chirp).
    Useful for frequency response analysis.
    """
    from scipy.signal import chirp
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')


def add_noise(signal: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add additive white Gaussian noise (AWGN) at a controlled SNR level.

    SNR (Signal-to-Noise Ratio):
        SNR_dB = 10 * log10(P_signal / P_noise)

    Args:
        signal: Input signal
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy signal
    """
    signal_power = np.mean(signal ** 2)
    snr_linear   = 10 ** (snr_db / 10)
    noise_power  = signal_power / snr_linear
    noise        = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def generate_composite_signal(duration: float = 2.0, fs: int = 16000) -> np.ndarray:
    """
    Generate a signal composed of multiple harmonics (spectral analysis demo).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Typical harmonics of a voiced speech signal
    signal = (
        0.50 * np.sin(2 * np.pi * 200  * t) +  # Fundamental
        0.30 * np.sin(2 * np.pi * 400  * t) +  # 2nd harmonic
        0.15 * np.sin(2 * np.pi * 800  * t) +  # 4th harmonic
        0.05 * np.sin(2 * np.pi * 1600 * t)    # 8th harmonic
    )
    return signal / np.max(np.abs(signal))
