"""
run_classifier.py
-----------------
End-to-end spoken digit classification analysis.

Dataset : Free Spoken Digit Dataset (FSDD) — synthetic equivalent
          500 recordings: 10 digits × 5 speakers × 10 utterances
          Files: data/fsdd/<digit>_<speaker>_<index>.wav

Classifiers: SVM (RBF), Random Forest, KNN, MLP, Gradient Boosting
Features:    MFCC (13) + Δ + ΔΔ + ZCR + RMS + spectral band energies (46 dims)
Outputs:     4 analysis panels in output/classifier/

Run: python run_classifier.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from src.classifier     import (load_dataset, build_classifiers,
                                 evaluate_classifiers, compute_learning_curve)
from src.classifier_viz import (plot_accuracy_comparison, plot_confusion_matrix,
                                 plot_f1_per_class, plot_learning_curves,
                                 plot_roc_curves, plot_feature_importance,
                                 plot_pca_scatter, save_figure)

OUT = "output/classifier"
os.makedirs(OUT, exist_ok=True)

DATA_DIR = "data/fsdd"
FS       = 16000


def banner(t):
    print(f"\n{'═'*62}\n  {t}\n{'═'*62}")


# ══════════════════════════════════════════════════════════════
# 1. Load dataset & extract features
# ══════════════════════════════════════════════════════════════
banner("STEP 1 – Loading Dataset & Extracting Features")
X, y, y_str, speakers, le = load_dataset(DATA_DIR, FS)
class_names = [str(c) for c in le.classes_]
n_classes   = len(class_names)

# Build feature names for importance plot
feat_names = []
for prefix in ['MFCC', 'Delta', 'Delta2']:
    feat_names += [f'{prefix}_mean_{i}' for i in range(13)]
    feat_names += [f'{prefix}_std_{i}'  for i in range(13)]
feat_names += ['ZCR_mean','ZCR_std','RMS_mean','RMS_std']
feat_names += ['Band_0-500','Band_500-1k','Band_1k-2k','Band_2k-4k','Band_4k-8k']

print(f"\n  Classes  : {class_names}")
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Speakers : {np.unique(speakers).tolist()}")


# ══════════════════════════════════════════════════════════════
# 2. Train & evaluate all classifiers
# ══════════════════════════════════════════════════════════════
banner("STEP 2 – Training & Evaluating Classifiers")
classifiers = build_classifiers()
results, X_train, X_test, y_train, y_test = evaluate_classifiers(
    X, y, classifiers, test_size=0.2, cv_folds=5)


# ══════════════════════════════════════════════════════════════
# 3. Learning curves (best 3 models)
# ══════════════════════════════════════════════════════════════
banner("STEP 3 – Computing Learning Curves")
top3 = sorted(results, key=lambda n: results[n]['accuracy'], reverse=True)[:3]
lc_data = {}
for name in top3:
    print(f"  Learning curve: {name}")
    sizes, tr, val = compute_learning_curve(classifiers[name], X, y)
    lc_data[name]  = (sizes, tr, val)


# ══════════════════════════════════════════════════════════════
# PANEL A: Overview — accuracy + confusion matrix (best model)
# ══════════════════════════════════════════════════════════════
banner("PANEL A – Accuracy Overview")

best_name = max(results, key=lambda n: results[n]['accuracy'])
best_cm   = results[best_name]['cm']

fig = plt.figure(figsize=(16, 7), facecolor='white')
fig.suptitle('Panel A – Classifier Accuracy & Best Model Confusion Matrix',
             fontsize=13, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(1, 2, wspace=0.4)

plot_accuracy_comparison(results, ax=fig.add_subplot(gs[0]))
plot_confusion_matrix(best_cm, class_names,
                      title=f'Confusion Matrix – {best_name}\n(normalized)',
                      ax=fig.add_subplot(gs[1]))
save_figure(fig, f"{OUT}/panelA_accuracy_overview.png")


# ══════════════════════════════════════════════════════════════
# PANEL B: Per-class F1 + ROC curves
# ══════════════════════════════════════════════════════════════
banner("PANEL B – F1 per Class & ROC Curves")

fig = plt.figure(figsize=(16, 7), facecolor='white')
fig.suptitle('Panel B – F1-Score per Digit & ROC Curves (Macro OvR)',
             fontsize=13, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(1, 2, wspace=0.4)

plot_f1_per_class(results, class_names, ax=fig.add_subplot(gs[0]))
plot_roc_curves(results, y_test, n_classes, ax=fig.add_subplot(gs[1]))
save_figure(fig, f"{OUT}/panelB_f1_roc.png")


# ══════════════════════════════════════════════════════════════
# PANEL C: Learning curves + PCA scatter
# ══════════════════════════════════════════════════════════════
banner("PANEL C – Learning Curves & Feature Space")

fig = plt.figure(figsize=(16, 7), facecolor='white')
fig.suptitle('Panel C – Learning Curves & PCA Feature Space',
             fontsize=13, fontweight='bold', y=1.01)
gs = gridspec.GridSpec(1, 2, wspace=0.4)

plot_learning_curves(lc_data, ax=fig.add_subplot(gs[0]))
plot_pca_scatter(X, y, class_names, ax=fig.add_subplot(gs[1]))
save_figure(fig, f"{OUT}/panelC_learning_pca.png")


# ══════════════════════════════════════════════════════════════
# PANEL D: Feature importance + all confusion matrices
# ══════════════════════════════════════════════════════════════
banner("PANEL D – Feature Importance & All Confusion Matrices")

fig = plt.figure(figsize=(18, 10), facecolor='white')
fig.suptitle('Panel D – Feature Importance (RF) & All Confusion Matrices',
             fontsize=13, fontweight='bold', y=1.01)

n_clf = len(results)
gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.45)

rf_pipe = classifiers["Random Forest"]
plot_feature_importance(rf_pipe, feat_names, top_n=15, ax=fig.add_subplot(gs[0, :]))

for idx, (name, res) in enumerate(results.items()):
    row, col = 1, idx % 3
    if idx < 3:
        ax = fig.add_subplot(gs[row, col])
        short = name.split()[0]
        plot_confusion_matrix(res['cm'], class_names,
                              title=f'{short} (acc={res["accuracy"]:.3f})',
                              ax=ax, normalize=True)

save_figure(fig, f"{OUT}/panelD_importance_matrices.png")


# ══════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════
banner("SUMMARY")
print(f"\n{'Model':<22} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>8}")
print("─" * 54)
for name in sorted(results, key=lambda n: results[n]['accuracy'], reverse=True):
    r = results[name]
    print(f"{name:<22} {r['accuracy']:>10.4f} {r['cv_mean']:>10.4f} {r['cv_std']:>8.4f}")

print(f"\n✅  All classifier panels saved to: ./{OUT}/")
