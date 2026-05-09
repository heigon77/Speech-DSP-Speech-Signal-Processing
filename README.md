# 🎙️ speech_dsp — Speech Signal Processing in Python

A complete, modular Python project covering the core techniques of **digital signal processing (DSP) applied to speech signals** — from real audio loading and filtering through spectral analysis, feature extraction (MFCCs), pitch detection, and voice activity detection (VAD).

---

## 📁 Project Structure

```
speech_dsp/
├── run_analysis.py          # DSP demo — loads real audio → 5 analysis panels
├── run_classifier.py        # Classifier demo — spoken-digit recognition
├── README.md
├── data/                    # ← put your WAV files here
│   └── *.wav                # any WAV files (FSDD format or plain)
└── src/
    ├── __init__.py
    ├── signal_generator.py  # Synthetic test signals (chirp, composite, noise)
    ├── filters.py           # Digital filters (IIR, FIR, pre-emphasis)
    ├── features.py          # Feature extraction (STFT, Mel, MFCC, ZCR, Pitch)
    ├── vad.py               # Voice Activity Detection
    ├── classifier.py        # ML classifiers + feature pipeline
    ├── classifier_viz.py    # Classifier visualisation utilities
    └── visualization.py     # Matplotlib-based plotting utilities
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install numpy scipy matplotlib soundfile scikit-learn
```

### 2. Add audio files

Place WAV files in the `data/` folder.  
`run_analysis.py` accepts any mono or stereo WAV (it converts to mono automatically).  
`run_classifier.py` expects FSDD-style naming: `<digit>_<speaker>_<index>.wav`.

### 3. Run the DSP analysis

```bash
python run_analysis.py                        # auto-detects first WAV in data/
python run_analysis.py --audio data/my.wav    # specify a file
python run_analysis.py --data  data/fsdd      # scan a sub-folder
```

Generates **5 PNG panels** inside `output/`:

| File | Content |
|---|---|
| `panel1_filtering.png` | Waveforms + digital filtering |
| `panel2_spectral.png` | FFT, Welch PSD, spectrograms |
| `panel3_mfcc.png` | Mel filterbank + MFCCs + Δ + ΔΔ |
| `panel4_pitch_vad.png` | F0 pitch track + VAD |
| `panel5_filters.png` | Filter frequency responses |

### 4. Run the classifier

```bash
python run_classifier.py   # expects WAV files in data/fsdd/
```

---

## 🔊 Audio Loading (`run_analysis.py`)

`run_analysis.py` loads real audio files from `data/` using the same pipeline as `run_classifier.py`:

| Step | Detail |
|---|---|
| **Read** | `soundfile.read()` — supports WAV, FLAC, OGG, … |
| **Mono** | multi-channel → averaged to mono |
| **Resample** | linear interpolation to 16 000 Hz if needed |
| **Trim / pad** | fixed 2 s analysis window (zero-padded if file is shorter) |
| **Normalise** | peak normalisation to ±0.9 |

> **Note:** Panel 2 still uses a generated composite signal and chirp sweep.  
> These are test signals designed to illustrate specific spectral phenomena  
> (harmonic structure and linear frequency sweep) — not speech recordings.

---

## 🧩 Modules

### `signal_generator.py` — Synthetic Test Signals

Used only for the composite/chirp demonstration in Panel 2.

| Function | Description |
|---|---|
| `generate_chirp()` | Linear frequency sweep |
| `generate_composite_signal()` | Multi-harmonic composite signal |
| `add_noise()` | AWGN at a controlled SNR (dB) |

---

### `filters.py` — Digital Filters

#### IIR — Butterworth

| Function | Description |
|---|---|
| `lowpass_filter()` | Lowpass filter |
| `highpass_filter()` | Highpass filter |
| `bandpass_filter()` | Bandpass filter |
| `bandstop_filter()` | Bandstop / notch filter |

All IIR filters use `filtfilt` (zero-phase, forward–backward filtering) to eliminate phase distortion.

#### FIR

| Function | Description |
|---|---|
| `fir_filter()` | Hamming-windowed FIR (lowpass/highpass/bandpass/bandstop) |

FIR filters guarantee **linear phase** (no group delay distortion) at the cost of higher order.

#### Speech Pre-processing

| Function | Description |
|---|---|
| `pre_emphasis()` | Pre-emphasis filter: `y[n] = x[n] − α·x[n−1]` (α ≈ 0.97) |
| `moving_average()` | Simple moving average smoother |
| `frequency_response()` | Frequency response (magnitude in dB + phase) |

**Pre-emphasis** compensates for the natural −6 dB/octave roll-off of the vocal tract, boosting high-frequency SNR before MFCC extraction — standard in ASR pipelines.

---

### `features.py` — Feature Extraction

#### Time-Frequency Analysis

| Function | Description |
|---|---|
| `stft()` | Short-Time Fourier Transform |
| `power_spectrogram()` | Power spectrogram in dB |
| `welch_psd()` | Welch PSD estimate |

**STFT parameters** follow speech conventions:
- Window: 25 ms (captures local stationarity)
- Hop: 10 ms (60% overlap)
- FFT size: next power of 2 ≥ window length

#### Mel Scale & Filterbank

| Function | Description |
|---|---|
| `mel_filterbank()` | Triangular Mel filterbank matrix |

The **Mel scale** approximates human auditory perception: fine resolution at low frequencies, coarse at high frequencies.

#### MFCC Pipeline

```
Signal → Pre-emphasis → STFT → Power Spectrum → Mel Filterbank → log(·) → DCT → MFCCs
```

| Function | Description |
|---|---|
| `mfcc()` | Full MFCC extraction (13 coefficients by default) |
| `delta()` | Delta (velocity) coefficients via regression |

#### Temporal Features

| Function | Description |
|---|---|
| `zero_crossing_rate()` | ZCR — discriminates voiced vs unvoiced frames |
| `rms_energy()` | Per-frame RMS energy |

#### Pitch (F0) Detection

| Function | Description |
|---|---|
| `detect_pitch()` | Autocorrelation-based fundamental frequency estimator |

---

### `vad.py` — Voice Activity Detection

| Function | Description |
|---|---|
| `energy_zcr_vad()` | Classic energy + ZCR VAD with smoothing |
| `adaptive_vad()` | Adaptive threshold VAD (exponential averaging) |
| `extract_voiced_segments()` | Extract voiced segments as (start, end, waveform) |

---

### `visualization.py` — Plotting Utilities

| Function | Renders |
|---|---|
| `plot_waveform()` | Time-domain waveform |
| `plot_spectrum()` | Frequency spectrum (dB) |
| `plot_spectrogram()` | Time-frequency spectrogram (inferno colormap) |
| `plot_mfcc()` | MFCC heatmap |
| `plot_mel_filterbank()` | Mel filterbank triangles |
| `plot_pitch()` | F0 trajectory scatter plot |
| `plot_vad()` | RMS energy + VAD regions |
| `plot_filter_response()` | Filter frequency response |

---

## 🔬 Key DSP Concepts Covered

| Concept | Where |
|---|---|
| Nyquist theorem & sampling | all modules |
| Spectral leakage & windowing | `features.py` |
| Zero-phase filtering | `filters.py → filtfilt` |
| IIR vs FIR trade-offs | `filters.py`, Panel 5 |
| Mel scale & auditory perception | `features.py → mel_filterbank()` |
| Cepstral analysis (MFCC) | `features.py → mfcc()` |
| Delta features (dynamics) | `features.py → delta()` |
| Pitch via autocorrelation | `features.py → detect_pitch()` |
| VAD — voiced/unvoiced segmentation | `vad.py` |
| Welch PSD estimation | `features.py → welch_psd()` |
| SNR & AWGN | `signal_generator.py → add_noise()` |

---

## 📚 References

- **Fant, G.** (1960). *Acoustic Theory of Speech Production*. Mouton.
- **Davis, S. & Mermelstein, P.** (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE TASLP*.
- **O'Shaughnessy, D.** (1987). *Speech Communication: Human and Machine*. Addison-Wesley.
- **Welch, P.D.** (1967). The use of FFT for the estimation of power spectra. *IEEE Trans. Audio Electroacoust.*
- **ITU-T G.729 Annex B** — VAD for low-bitrate speech coding.
- **Rabiner, L. & Schafer, R.** (2010). *Theory and Applications of Digital Speech Processing*. Prentice Hall.

---

## 🛠️ Dependencies

| Library | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Array operations, FFT |
| `scipy` | ≥ 1.10 | Filters, signal processing, DCT |
| `matplotlib` | ≥ 3.7 | Visualization |
| `soundfile` | ≥ 0.12 | WAV / FLAC / OGG reading |
| `scikit-learn` | ≥ 1.3 | Classifiers (`run_classifier.py`) |

---

## 📄 License

MIT License — free to use, modify, and distribute.
