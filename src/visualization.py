"""
visualization.py
----------------
Plotting utilities for speech signal analysis.
Uses Matplotlib with a clean scientific style.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


COLORS = {
    'signal':    '#2196F3',
    'filtered':  '#4CAF50',
    'energy':    '#FF5722',
    'pitch':     '#9C27B0',
    'voice':     '#4CAF50',
    'silence':   '#F44336',
    'background':'#FAFAFA',
    'grid':      '#E0E0E0',
}


def _apply_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor':   '#F8F9FA',
        'axes.grid':        True,
        'grid.color':       COLORS['grid'],
        'grid.linewidth':   0.5,
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'font.family':      'DejaVu Sans',
        'axes.titlesize':   11,
        'axes.labelsize':   9,
        'xtick.labelsize':  8,
        'ytick.labelsize':  8,
    })


def plot_waveform(signal: np.ndarray, fs: int = 16000,
                  title: str = 'Waveform', ax=None,
                  color: str = None) -> plt.Axes:
    """Plot the time-domain waveform."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, color=color or COLORS['signal'], linewidth=0.7, alpha=0.9)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.set_xlim(0, t[-1])
    return ax


def plot_spectrum(freqs: np.ndarray, magnitude_db: np.ndarray,
                  title: str = 'Power Spectrum',
                  ax=None, f_max: float = 8000) -> plt.Axes:
    """Plot the power spectrum in dB."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    mask = freqs <= f_max
    ax.plot(freqs[mask], magnitude_db[mask],
            color=COLORS['signal'], linewidth=1.2)
    ax.fill_between(freqs[mask], magnitude_db[mask],
                    magnitude_db[mask].min(), alpha=0.2, color=COLORS['signal'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    return ax


def plot_spectrogram(freqs: np.ndarray, times: np.ndarray,
                     spectrum_db: np.ndarray, title: str = 'Spectrogram',
                     ax=None, f_max: float = 8000) -> plt.Axes:
    """Plot a spectrogram (time x frequency x power in dB)."""
    _apply_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    freq_mask = freqs <= f_max
    vmin = np.percentile(spectrum_db, 10)
    vmax = np.percentile(spectrum_db, 99)

    img = ax.pcolormesh(times, freqs[freq_mask],
                        spectrum_db[:, freq_mask].T,
                        cmap='inferno', shading='gouraud',
                        norm=Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(img, ax=ax, label='dB', pad=0.01)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    return ax


def plot_mfcc(mfcc_coeffs: np.ndarray, hop_ms: float = 10.0,
              title: str = 'MFCC', ax=None) -> plt.Axes:
    """Plot MFCC coefficients over time as a heatmap."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    n_frames, n_coeffs = mfcc_coeffs.shape
    times = np.arange(n_frames) * hop_ms / 1000

    img = ax.pcolormesh(times, np.arange(n_coeffs),
                        mfcc_coeffs.T, cmap='RdBu_r', shading='gouraud')
    plt.colorbar(img, ax=ax, label='Value', pad=0.01)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_title(title)
    ax.set_yticks(np.arange(0, n_coeffs, 2))
    return ax


def plot_mel_filterbank(filterbank: np.ndarray, freqs: np.ndarray,
                        ax=None) -> plt.Axes:
    """Plot the triangular Mel filterbank."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    n_filters = filterbank.shape[0]
    cmap = plt.cm.viridis
    for i in range(n_filters):
        color = cmap(i / n_filters)
        ax.plot(freqs, filterbank[i], color=color, linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Weight')
    ax.set_title(f'Mel Filter Bank ({n_filters} filters)')
    ax.set_xlim(0, freqs[-1])
    return ax


def plot_pitch(times: np.ndarray, f0: np.ndarray,
               title: str = 'Fundamental Frequency (F0 / Pitch)',
               ax=None) -> plt.Axes:
    """Plot the pitch trajectory over time."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    voiced   = f0 > 0
    unvoiced = ~voiced
    ax.scatter(times[voiced],   f0[voiced],                   s=4, color=COLORS['pitch'],  zorder=3, label='Voiced')
    ax.scatter(times[unvoiced], np.zeros(np.sum(unvoiced)),   s=2, color='#BDBDBD', alpha=0.4, label='Unvoiced')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('F0 (Hz)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_vad(frame_times: np.ndarray, rms: np.ndarray,
             mask: np.ndarray, ax=None) -> plt.Axes:
    """Plot RMS energy with VAD regions highlighted."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 2.5))

    ax.plot(frame_times, rms, color='#607D8B', linewidth=1, label='RMS')

    n = len(mask)
    for i in range(n):
        if i < len(frame_times):
            color = COLORS['voice'] if mask[i] else '#F5F5F5'
            alpha = 0.3 if mask[i] else 0.0
            dt    = frame_times[1] - frame_times[0] if n > 1 else 0.01
            ax.axvspan(frame_times[i], frame_times[i] + dt, color=color, alpha=alpha)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMS')
    ax.set_title('VAD – Voice Activity Detection')

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=COLORS['voice'],  alpha=0.4, label='Voice'),
               Patch(facecolor='#E0E0E0', alpha=0.6, label='Silence')]
    ax.legend(handles=handles, fontsize=8)
    return ax


def plot_filter_response(freqs: np.ndarray, magnitude_db: np.ndarray,
                         title: str = 'Filter Frequency Response',
                         ax=None) -> plt.Axes:
    """Plot the frequency response of a digital filter."""
    _apply_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(freqs, magnitude_db, color=COLORS['filtered'], linewidth=1.5)
    ax.axhline(-3, color='red', linestyle='--', linewidth=1, label='-3 dB')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=max(magnitude_db.min(), -80))
    return ax


def save_figure(fig: plt.Figure, path: str, dpi: int = 150):
    """Save a figure to disk in high quality."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
