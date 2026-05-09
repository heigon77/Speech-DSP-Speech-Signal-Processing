"""
run_analysis.py
---------------
Full demo of the speech_dsp project.

Loads real audio from the data/ directory and generates 5 analysis panels:
  1. Signals and filtering
  2. Spectral analysis (FFT, Welch, Spectrogram)
  3. Mel filterbank + MFCCs + Delta coefficients
  4. Pitch (F0) + VAD
  5. Filter frequency response comparison

Audio file convention (same as run_classifier.py):
  data/<label>_<speaker>_<index>.wav   (FSDD-style)
  — or any *.wav file inside data/

Usage:
  python run_analysis.py                        # auto-detects first WAV in data/
  python run_analysis.py --audio data/myfile.wav
  python run_analysis.py --data  data/fsdd      # scan a sub-folder
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from src import *

import soundfile as sf          # same library used by run_classifier.py


# ── Constants ─────────────────────────────────────────────────────────────────
FS       = 16000          # target sampling rate (Hz)
DURATION = 2.0            # analysis window (s)
DATA_DIR = "data"         # default folder scanned for WAV files
OUT_DIR  = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print('═'*60)


def find_wav_files(data_dir: str) -> list:
    """Return sorted list of .wav paths found inside *data_dir*."""
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted(p.rglob("*.wav")) + sorted(p.rglob("*.WAV"))


def load_audio(path: str,
               target_fs: int = FS,
               duration: float = DURATION) -> np.ndarray:
    """
    Load a WAV file (same logic as load_dataset in src/classifier.py):
      • convert multi-channel to mono by averaging channels
      • resample via linear interpolation if sample rate differs
      • trim to *duration* seconds (or zero-pad if the file is shorter)
      • peak-normalise to ±0.9

    Args:
        path:      Path to the WAV file.
        target_fs: Target sampling rate in Hz.
        duration:  Desired signal length in seconds (None = keep full file).

    Returns:
        1-D NumPy float64 array.
    """
    audio, file_fs = sf.read(str(path))

    # Mono conversion
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed (mirrors run_classifier.py)
    if file_fs != target_fs:
        n_new = int(len(audio) * target_fs / file_fs)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_new),
            np.arange(len(audio)),
            audio,
        )

    # Trim / pad to requested duration
    if duration is not None:
        n_target = int(target_fs * duration)
        if len(audio) >= n_target:
            audio = audio[:n_target]
        else:
            audio = np.pad(audio, (0, n_target - len(audio)))

    # Peak normalisation
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9

    return audio


def resolve_audio_file(args) -> str:
    """
    Pick the audio file to use, in priority order:
      1. --audio flag
      2. first WAV found in --data dir (or DATA_DIR)
    Exits with an informative message if nothing is found.
    """
    if args.audio:
        path = Path(args.audio)
        if not path.is_file():
            sys.exit(f"[ERROR] Audio file not found: {args.audio}")
        return str(path)

    data_dir = args.data or DATA_DIR
    wavs = find_wav_files(data_dir)
    if not wavs:
        sys.exit(
            f"[ERROR] No WAV files found in '{data_dir}'.\n"
            "  Place audio files there or use:  python run_analysis.py --audio path/to/file.wav"
        )

    chosen = str(wavs[0])
    print(f"  Audio source : {chosen}  ({len(wavs)} file(s) available in '{data_dir}')")
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 1 – Signals and Filtering
# ══════════════════════════════════════════════════════════════════════════════
def panel_signals_filtering(audio_path: str):
    banner("PANEL 1 – Signals and Filtering")

    voice    = load_audio(audio_path)
    noisy    = add_noise(voice, snr_db=15)
    lpf      = lowpass_filter (noisy,  3000,       FS, order=5)
    hpf      = highpass_filter(noisy,   300,       FS, order=5)
    bpf      = bandpass_filter(noisy,   300, 3400, FS, order=4)
    emphasis = pre_emphasis(voice)

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.suptitle('Panel 1 – Speech Signals and Digital Filtering',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)

    plot_waveform(voice,    FS, 'Input Voice (clean)',                  ax=fig.add_subplot(gs[0, 0]))
    plot_waveform(noisy,    FS, 'Voice + White Noise (SNR = 15 dB)',   ax=fig.add_subplot(gs[0, 1]), color='#EF5350')
    plot_waveform(lpf,      FS, 'Butterworth Lowpass (fc = 3 kHz)',    ax=fig.add_subplot(gs[1, 0]), color='#4CAF50')
    plot_waveform(hpf,      FS, 'Butterworth Highpass (fc = 300 Hz)',  ax=fig.add_subplot(gs[1, 1]), color='#FF9800')
    plot_waveform(bpf,      FS, 'Bandpass 300–3400 Hz (telephone)',    ax=fig.add_subplot(gs[2, 0]), color='#9C27B0')
    plot_waveform(emphasis, FS, 'Pre-emphasis (α = 0.97)',             ax=fig.add_subplot(gs[2, 1]), color='#00BCD4')

    save_figure(fig, f"{OUT_DIR}/panel1_filtering.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 2 – Spectral Analysis
# ══════════════════════════════════════════════════════════════════════════════
def panel_spectral_analysis(audio_path: str):
    banner("PANEL 2 – Spectral Analysis")

    voice     = load_audio(audio_path)
    # Composite & chirp remain generated because they are test signals
    # designed to illustrate specific spectral phenomena (harmonics / sweep),
    # not speech recordings.
    composite = generate_composite_signal(DURATION, FS)
    chirp     = generate_chirp(100, 4000, DURATION, FS)

    # Direct FFT
    from scipy.fft import fft, fftfreq
    n         = len(voice)
    freqs_fft = fftfreq(n, 1 / FS)[:n // 2]
    mag_voice = 20 * np.log10(np.abs(fft(voice))[:n // 2] / n + 1e-9)
    mag_comp  = 20 * np.log10(np.abs(fft(composite))[:n // 2] / n + 1e-9)

    # Welch PSD
    freq_w, psd_voice = welch_psd(voice, FS)

    # Spectrograms
    freqs_s, times_s, spec_db_voice = power_spectrogram(voice,  FS)
    freqs_c, times_c, spec_db_chirp = power_spectrogram(chirp,  FS)

    fig = plt.figure(figsize=(14, 11), facecolor='white')
    fig.suptitle('Panel 2 – Spectral Analysis', fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)

    plot_spectrum(freqs_fft, mag_voice, 'FFT – Voice Spectrum',              ax=fig.add_subplot(gs[0, 0]))
    plot_spectrum(freqs_fft, mag_comp,  'FFT – Composite Signal (harmonics)',ax=fig.add_subplot(gs[0, 1]), f_max=4000)
    plot_spectrum(freq_w,    psd_voice, 'Welch PSD – Voice',                 ax=fig.add_subplot(gs[1, 0]))
    plot_spectrogram(freqs_s, times_s, spec_db_voice, 'Spectrogram – Voice',ax=fig.add_subplot(gs[1, 1]))
    plot_waveform(chirp, FS, 'Chirp (100–4000 Hz linear sweep)',             ax=fig.add_subplot(gs[2, 0]), color='#FF5722')
    plot_spectrogram(freqs_c, times_c, spec_db_chirp, 'Spectrogram – Chirp',ax=fig.add_subplot(gs[2, 1]))

    save_figure(fig, f"{OUT_DIR}/panel2_spectral.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 3 – Mel Filterbank, MFCCs, and Delta Coefficients
# ══════════════════════════════════════════════════════════════════════════════
def panel_mfcc(audio_path: str):
    banner("PANEL 3 – Mel Filterbank, MFCC, and Dynamic Coefficients")

    voice = load_audio(audio_path)

    # Mel filterbank
    n_fft     = 512
    n_filters = 26
    fb        = mel_filterbank(n_filters, n_fft, FS)
    freqs_mel = np.linspace(0, FS // 2, n_fft // 2 + 1)

    # MFCCs and delta coefficients
    mfcc_coeffs = mfcc(voice, FS, n_mfcc=13, n_filters=n_filters)
    delta1      = delta(mfcc_coeffs, N=2)
    delta2      = delta(delta1, N=2)

    fig = plt.figure(figsize=(14, 12), facecolor='white')
    fig.suptitle('Panel 3 – Mel Filterbank, MFCC, and Dynamic Coefficients',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.35)

    plot_mel_filterbank(fb, freqs_mel,                ax=fig.add_subplot(gs[0, :]))
    plot_mfcc(mfcc_coeffs, title='MFCCs (13 coefficients)',       ax=fig.add_subplot(gs[1, :]))
    plot_mfcc(delta1,      title='Delta-MFCC (Δ) – Velocity',    ax=fig.add_subplot(gs[2, 0]))
    plot_mfcc(delta2,      title='Delta-Delta-MFCC (ΔΔ) – Accel',ax=fig.add_subplot(gs[2, 1]))

    save_figure(fig, f"{OUT_DIR}/panel3_mfcc.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 4 – Pitch and VAD
# ══════════════════════════════════════════════════════════════════════════════
def panel_pitch_vad(audio_path: str):
    banner("PANEL 4 – Pitch (F0) and VAD")

    voice   = load_audio(audio_path)
    silence = np.zeros(int(0.3 * FS))

    # Signal with alternating voiced/silent segments
    seg_signal = np.concatenate([
        silence,
        voice[:int(0.5 * FS)],
        silence,
        voice[int(0.5 * FS):int(1.0 * FS)],
        silence,
    ])
    seg_signal = add_noise(seg_signal, snr_db=20)

    # Pitch
    times_p, f0 = detect_pitch(seg_signal, FS)

    # VAD
    frame_size  = int(FS * 0.025)
    hop         = int(FS * 0.010)
    n_frames    = 1 + (len(seg_signal) - frame_size) // hop
    frame_times = np.arange(n_frames) * hop / FS

    rms_arr, _, vad_mask           = energy_zcr_vad(seg_signal, FS)
    _,       adapt_mask, adapt_thr = adaptive_vad(seg_signal, FS)

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.suptitle('Panel 4 – Pitch (F0) Detection and VAD',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 1, hspace=0.55)

    ax1 = fig.add_subplot(gs[0])
    plot_waveform(seg_signal, FS, 'Segmented Signal (voice + silences + noise)', ax=ax1)

    ax2 = fig.add_subplot(gs[1])
    plot_pitch(times_p, f0, ax=ax2)

    ax3 = fig.add_subplot(gs[2])
    n   = min(len(frame_times), len(rms_arr), len(vad_mask))
    plot_vad(frame_times[:n], rms_arr[:n], vad_mask[:n], ax=ax3)

    n_thr = min(len(frame_times), len(adapt_thr))
    ax3.plot(frame_times[:n_thr], adapt_thr[:n_thr], '--',
             color='#FF5722', linewidth=1.2, label='Adaptive threshold')
    ax3.legend(fontsize=8)

    save_figure(fig, f"{OUT_DIR}/panel4_pitch_vad.png")


# ══════════════════════════════════════════════════════════════════════════════
# PANEL 5 – Filter Frequency Response Comparison  (no audio needed)
# ══════════════════════════════════════════════════════════════════════════════
def panel_filter_comparison():
    banner("PANEL 5 – Filter Comparison")

    from scipy.signal import butter, firwin, freqz

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    fig.suptitle('Panel 5 – Filter Frequency Responses',
                 fontsize=13, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.35)

    # Butterworth lowpass: effect of order
    ax1 = fig.add_subplot(gs[0, 0])
    for order in [1, 3, 5, 9]:
        b, a = butter(order, 3000 / (FS / 2), btype='low')
        f, m, _ = frequency_response(b, a, FS)
        ax1.plot(f, m, label=f'Order {order}', linewidth=1.5)
    ax1.axhline(-3, color='red', linestyle='--', linewidth=1, label='-3 dB')
    ax1.set_title('Butterworth LP – Effect of Order')
    ax1.set_xlabel('Frequency (Hz)'); ax1.set_ylabel('dB')
    ax1.set_ylim(-80, 5); ax1.set_xlim(0, 8000)
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # Highpass with different cutoffs
    ax2 = fig.add_subplot(gs[0, 1])
    for fc in [200, 500, 1000, 2000]:
        b, a = butter(5, fc / (FS / 2), btype='high')
        f, m, _ = frequency_response(b, a, FS)
        ax2.plot(f, m, label=f'fc = {fc} Hz', linewidth=1.5)
    ax2.axhline(-3, color='red', linestyle='--', linewidth=1)
    ax2.set_title('Butterworth HP – Different Cutoffs')
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('dB')
    ax2.set_ylim(-80, 5); ax2.set_xlim(0, 8000)
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # IIR vs FIR
    ax3 = fig.add_subplot(gs[1, 0])
    b_iir, a_iir = butter(5, 3000 / (FS / 2), btype='low')
    f_iir, m_iir, _ = frequency_response(b_iir, a_iir, FS)

    taps51,  = firwin(51,  3000 / (FS / 2), window='hamming'),
    w51, h51   = freqz(taps51,  worN=512)
    taps101, = firwin(101, 3000 / (FS / 2), window='hamming'),
    w101, h101 = freqz(taps101, worN=512)

    ax3.plot(f_iir,                    m_iir,                                label='IIR Butterworth (ord=5)', linewidth=1.5, color='#2196F3')
    ax3.plot(w51  * FS / (2*np.pi), 20*np.log10(np.abs(h51)  + 1e-9),       label='FIR 51 taps',             linewidth=1.5, color='#4CAF50')
    ax3.plot(w101 * FS / (2*np.pi), 20*np.log10(np.abs(h101) + 1e-9),       label='FIR 101 taps',            linewidth=1.5, color='#FF5722')
    ax3.axhline(-3, color='red', linestyle='--', linewidth=1)
    ax3.set_title('IIR vs FIR (Lowpass 3 kHz)')
    ax3.set_xlabel('Frequency (Hz)'); ax3.set_ylabel('dB')
    ax3.set_ylim(-80, 5); ax3.set_xlim(0, 8000)
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # Notch filter (60 Hz power-line)
    ax4 = fig.add_subplot(gs[1, 1])
    b_n, a_n = butter(4, [55 / (FS / 2), 65 / (FS / 2)], btype='bandstop')
    f_n, m_n, _ = frequency_response(b_n, a_n, FS)
    ax4.plot(f_n, m_n, color='#9C27B0', linewidth=1.5)
    ax4.axvline(60, color='red', linestyle='--', linewidth=1, label='60 Hz')
    ax4.set_title('Notch Filter – 60 Hz Rejection (power line)')
    ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('dB')
    ax4.set_ylim(-80, 5); ax4.set_xlim(0, 500)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    save_figure(fig, f"{OUT_DIR}/panel5_filters.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Speech DSP analysis — loads real audio from data/")
    parser.add_argument(
        "--audio", metavar="FILE",
        help="Path to a specific WAV file (overrides --data)")
    parser.add_argument(
        "--data", metavar="DIR", default=DATA_DIR,
        help=f"Directory to scan for WAV files (default: {DATA_DIR})")
    return parser.parse_args()


if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════╗")
    print("║     SPEECH DSP – Speech Signal Processing in Python  ║")
    print("╚══════════════════════════════════════════════════════╝")

    args = parse_args()
    audio_path = resolve_audio_file(args)

    print(f"\n  File     : {audio_path}")
    print(f"  Fs target: {FS} Hz  |  Window: {DURATION} s")

    panel_signals_filtering(audio_path)
    panel_spectral_analysis(audio_path)
    panel_mfcc(audio_path)
    panel_pitch_vad(audio_path)
    panel_filter_comparison()

    print(f"\n✅  All panels saved to: ./{OUT_DIR}/")
    print("   panel1_filtering.png")
    print("   panel2_spectral.png")
    print("   panel3_mfcc.png")
    print("   panel4_pitch_vad.png")
    print("   panel5_filters.png")
