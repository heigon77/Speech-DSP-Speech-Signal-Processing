"""
classifier_viz.py
-----------------
Visualization functions for classifier analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import itertools


PALETTE = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0',
           '#00BCD4','#FF5722','#607D8B','#8BC34A','#FFC107']


def _style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': '#F8F9FA',
        'axes.grid': True, 'grid.color': '#E0E0E0', 'grid.linewidth': 0.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.family': 'DejaVu Sans', 'axes.titlesize': 11,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    })


# ── Panel A: accuracy comparison bar chart ─────────────────────────────────────
def plot_accuracy_comparison(results: dict, ax=None) -> plt.Axes:
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    names  = list(results.keys())
    accs   = [results[n]["accuracy"]  for n in names]
    cv_m   = [results[n]["cv_mean"]   for n in names]
    cv_s   = [results[n]["cv_std"]    for n in names]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs,  w, label='Test Accuracy',
                   color=PALETTE[:len(names)], alpha=0.85, zorder=3)
    bars2 = ax.bar(x + w/2, cv_m,  w, label='CV Accuracy (mean)',
                   color=PALETTE[:len(names)], alpha=0.45, zorder=3,
                   yerr=cv_s, capsize=4, error_kw=dict(elinewidth=1))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title('Classifier Accuracy Comparison')
    ax.legend(fontsize=8)
    ax.axhline(1/10, color='red', linestyle='--', linewidth=1,
               label='Random baseline (10 classes)')

    for bar, v in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    return ax


# ── Panel B: confusion matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, class_names: list,
                          title: str = 'Confusion Matrix',
                          ax=None, normalize: bool = True) -> plt.Axes:
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if normalize:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = '.2f'
    else:
        cm_plot = cm
        fmt = 'd'

    cmap = LinearSegmentedColormap.from_list('cm_cmap', ['#FFFFFF','#1565C0'])
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=cmap,
                   vmin=0, vmax=1 if normalize else cm.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm_plot.max() / 2
    for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
        val = f'{cm_plot[i,j]:{fmt}}'
        ax.text(j, i, val, ha='center', va='center', fontsize=7,
                color='white' if cm_plot[i,j] > thresh else 'black')

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    return ax


# ── Panel C: per-class F1 score ────────────────────────────────────────────────
def plot_f1_per_class(results: dict, class_names: list, ax=None) -> plt.Axes:
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(class_names))
    n = len(results)
    width = 0.8 / n

    for i, (name, res) in enumerate(results.items()):
        f1s = [res["report"].get(str(c), {}).get('f1-score', 0)
               for c in range(len(class_names))]
        offset = (i - n/2 + 0.5) * width
        ax.bar(x + offset, f1s, width, label=name,
               color=PALETTE[i % len(PALETTE)], alpha=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f'"{c}"' for c in class_names])
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('F1-Score per Digit Class')
    ax.legend(fontsize=7, loc='lower right')
    return ax


# ── Panel D: learning curves ───────────────────────────────────────────────────
def plot_learning_curves(lc_data: dict, ax=None) -> plt.Axes:
    """
    lc_data: {name: (train_sizes, train_scores, val_scores)}
    """
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for i, (name, (sizes, tr_sc, val_sc)) in enumerate(lc_data.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(sizes, val_sc.mean(1), '-o', color=color,
                label=name, linewidth=1.8, markersize=4)
        ax.fill_between(sizes,
                        val_sc.mean(1) - val_sc.std(1),
                        val_sc.mean(1) + val_sc.std(1),
                        alpha=0.12, color=color)

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Learning Curves (Validation)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    return ax


# ── Panel E: ROC curves (macro-OvR) ───────────────────────────────────────────
def plot_roc_curves(results: dict, y_test: np.ndarray,
                    n_classes: int, ax=None) -> plt.Axes:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    y_bin = label_binarize(y_test, classes=np.arange(n_classes))

    for i, (name, res) in enumerate(results.items()):
        if res["y_prob"] is None:
            continue
        # Macro-average ROC
        fpr_all, tpr_all = [], []
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, c], res["y_prob"][:, c])
            fpr_all.append(fpr); tpr_all.append(tpr)
        all_fpr = np.unique(np.concatenate(fpr_all))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_all[c], tpr_all[c])
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color=PALETTE[i % len(PALETTE)],
                linewidth=1.8, label=f'{name} (AUC={roc_auc:.3f})')

    ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – Macro Average (One-vs-Rest)')
    ax.legend(fontsize=8)
    return ax


# ── Panel F: feature importance (Random Forest) ───────────────────────────────
def plot_feature_importance(clf_pipeline, feature_names: list,
                            top_n: int = 20, ax=None) -> plt.Axes:
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    rf = clf_pipeline.named_steps['clf']
    if not hasattr(rf, 'feature_importances_'):
        ax.text(0.5, 0.5, 'Feature importances not available',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
    ax.barh(range(top_n), vals[::-1], color=colors, alpha=0.85, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=7)
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Features – Random Forest')
    return ax


# ── Panel G: PCA scatter plot ──────────────────────────────────────────────────
def plot_pca_scatter(X: np.ndarray, y: np.ndarray,
                     class_names: list, ax=None) -> plt.Axes:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    X_scaled = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    var  = pca.explained_variance_ratio_

    for c_idx, name in enumerate(class_names):
        mask = y == c_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=18, alpha=0.6, color=PALETTE[c_idx % len(PALETTE)],
                   label=f'"{name}"', zorder=3)

    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% variance)')
    ax.set_title('PCA – 2D Feature Space (MFCC + Delta + ZCR + RMS)')
    ax.legend(fontsize=7, ncol=2, loc='best')
    return ax


def save_figure(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")
