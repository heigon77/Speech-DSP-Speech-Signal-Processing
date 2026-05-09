"""
speech_dsp — Speech Signal Processing in Python
"""
from .signal_generator import (
    generate_pure_tone, generate_synthetic_voice, generate_chirp,
    add_noise, generate_composite_signal
)
from .filters import (
    lowpass_filter, highpass_filter, bandpass_filter,
    bandstop_filter, fir_filter, pre_emphasis, moving_average,
    frequency_response
)
from .features import (
    stft, power_spectrogram, mel_filterbank,
    mfcc, delta, zero_crossing_rate, rms_energy,
    detect_pitch, welch_psd
)
from .vad import energy_zcr_vad, adaptive_vad, extract_voiced_segments
from .visualization import (
    plot_waveform, plot_spectrum, plot_spectrogram,
    plot_mfcc, plot_mel_filterbank, plot_pitch, plot_vad,
    plot_filter_response, save_figure
)
from .classifier import (
    extract_features, load_dataset, build_classifiers,
    evaluate_classifiers, compute_learning_curve
)
from .classifier_viz import (
    plot_accuracy_comparison, plot_confusion_matrix, plot_f1_per_class,
    plot_learning_curves, plot_roc_curves, plot_feature_importance,
    plot_pca_scatter
)
