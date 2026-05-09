"""
classifier.py
-------------
Audio classifier pipeline for spoken digit recognition.

Pipeline:
  1. Load WAV files + extract labels from filename
  2. Feature extraction: MFCC + Delta + Delta-Delta + ZCR + RMS
  3. Train & evaluate: SVM, Random Forest, KNN, MLP, Gradient Boosting
  4. Metrics: accuracy, confusion matrix, classification report, ROC-AUC
  5. Cross-validation + learning curves
"""

import os
import numpy as np
import soundfile as sf
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, learning_curve)
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from .features import mfcc, delta, zero_crossing_rate, rms_energy, welch_psd


# ─────────────────────────── FEATURE EXTRACTION ────────────────────────────────

def extract_features(audio: np.ndarray, fs: int = 16000,
                     n_mfcc: int = 13) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a raw audio array.

    Feature set (total = 13*3 + 2 + 5 = 46 dimensions):
      - 13 MFCCs          → mean + std over time
      - 13 Delta-MFCCs    → mean + std
      - 13 Delta-Delta    → mean + std
      - ZCR               → mean + std
      - RMS energy        → mean + std
      - 5 Welch PSD bands → log-energy in 5 perceptual bands

    Aggregation via statistics (mean/std) converts variable-length
    audio into a fixed-size vector suitable for classic ML models.
    """
    # Pad/trim to consistent length
    target = int(fs * 1.0)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]

    # ── MFCCs ──────────────────────────────────────────────────────
    mfcc_c  = mfcc(audio, fs, n_mfcc=n_mfcc)
    delta1  = delta(mfcc_c)
    delta2  = delta(delta1)

    mfcc_feat  = np.concatenate([mfcc_c.mean(0),  mfcc_c.std(0)])
    delta1_feat = np.concatenate([delta1.mean(0), delta1.std(0)])
    delta2_feat = np.concatenate([delta2.mean(0), delta2.std(0)])

    # ── ZCR & RMS ──────────────────────────────────────────────────
    frame_sz = int(fs * 0.025)
    hop      = int(fs * 0.010)
    zcr_arr  = zero_crossing_rate(audio, frame_sz, hop)
    rms_arr  = rms_energy(audio, frame_sz, hop)
    zcr_feat = np.array([zcr_arr.mean(), zcr_arr.std()])
    rms_feat = np.array([rms_arr.mean(), rms_arr.std()])

    # ── Spectral band energies (Welch) ─────────────────────────────
    freqs, psd_db = welch_psd(audio, fs)
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_feat = []
    for f_lo, f_hi in bands:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        band_feat.append(np.mean(psd_db[mask]) if mask.any() else -80.0)
    band_feat = np.array(band_feat)

    return np.concatenate([mfcc_feat, delta1_feat, delta2_feat,
                           zcr_feat, rms_feat, band_feat])


def load_dataset(data_dir: str, fs: int = 16000) -> tuple:
    """
    Load all WAV files from data_dir.
    Filename convention: <label>_<speaker>_<index>.wav

    Returns:
        X       : feature matrix (n_samples x n_features)
        y       : integer labels
        labels  : string labels
        speakers: speaker IDs per sample
        le      : fitted LabelEncoder
    """
    paths    = sorted(Path(data_dir).glob("*.wav"))
    X_list, y_list, speakers = [], [], []

    print(f"  Loading {len(paths)} audio files...")
    for i, path in enumerate(paths):
        parts   = path.stem.split('_')
        label   = parts[0]
        speaker = parts[1] if len(parts) > 1 else 'unknown'

        audio, file_fs = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_fs != fs:
            # Simple resampling by interpolation
            n_new = int(len(audio) * fs / file_fs)
            audio = np.interp(np.linspace(0, len(audio)-1, n_new),
                              np.arange(len(audio)), audio)

        feat = extract_features(audio, fs)
        X_list.append(feat)
        y_list.append(label)
        speakers.append(speaker)

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(paths)} files processed")

    X  = np.array(X_list)
    le = LabelEncoder()
    y  = le.fit_transform(y_list)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(le.classes_)} classes")
    return X, y, np.array(y_list), np.array(speakers), le


# ─────────────────────────── CLASSIFIERS ───────────────────────────────────────

def build_classifiers() -> dict:
    """
    Return a dictionary of sklearn classifier pipelines.
    Each pipeline: StandardScaler → (optional PCA) → Classifier.

    Models chosen to demonstrate diversity:
      - SVM:  strong with high-dimensional acoustic features (kernel trick)
      - RF:   ensemble, robust to irrelevant features
      - KNN:  non-parametric, interpretable distance in MFCC space
      - MLP:  shallow neural network (hidden layers)
      - GBM:  gradient boosting, handles feature interactions well
    """
    return {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel='rbf', C=10, gamma='scale',
                          probability=True, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              random_state=42, n_jobs=-1))
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=30, random_state=42)),
            ("clf",    KNeighborsClassifier(n_neighbors=5, metric='euclidean'))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    MLPClassifier(hidden_layer_sizes=(128, 64),
                                     activation='relu', max_iter=500,
                                     random_state=42, early_stopping=True))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=150,
                                                   learning_rate=0.1,
                                                   max_depth=4,
                                                   random_state=42))
        ]),
    }


# ─────────────────────────── EVALUATION ────────────────────────────────────────

def evaluate_classifiers(X, y, classifiers: dict,
                          test_size: float = 0.2,
                          cv_folds: int = 5) -> dict:
    """
    Train and evaluate all classifiers.

    Returns a results dict with:
      accuracy, cv_mean, cv_std, confusion_matrix,
      classification_report, y_pred, y_prob (for ROC)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)

    cv      = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, clf in classifiers.items():
        print(f"\n  Training: {name}")

        # Cross-validation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                     scoring='accuracy', n_jobs=-1)

        # Final fit on full train set
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy":   acc,
            "cv_mean":    cv_scores.mean(),
            "cv_std":     cv_scores.std(),
            "cm":         cm,
            "report":     report,
            "y_pred":     y_pred,
            "y_prob":     y_prob,
            "clf":        clf,
        }
        print(f"    Test accuracy: {acc:.4f} | "
              f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return results, X_train, X_test, y_train, y_test


def compute_learning_curve(clf, X, y, cv_folds: int = 5) -> tuple:
    """
    Compute learning curve (train/val accuracy vs training set size).
    Useful to diagnose bias/variance trade-off.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1
    )
    return train_sizes, train_scores, val_scores
