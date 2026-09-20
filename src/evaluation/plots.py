"""Visualization and plotting for evaluation results."""
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,
)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
FIGSIZE = (8, 6)
DPI = 150


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    labels: List[str], save_path: str, title: str = "Confusion Matrix",
):
    """Annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=True, square=True,
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(title, fontsize=14, weight="bold")
    
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.75, f"({pct:.1f}%)", ha="center", va="center",
                    fontsize=10, color="gray")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    save_path: str, title: str = "ROC Curve",
):
    """Plot ROC curve with AUC and diagonal reference."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    youdens_j = tpr - fpr
    optimal_idx = np.argmax(youdens_j)
    
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], "ro", markersize=10,
            label=f"Optimal (J={youdens_j[optimal_idx]:.3f}, t={thresholds[optimal_idx]:.3f})")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    save_path: str, title: str = "Precision-Recall Curve",
):
    """Plot Precision-Recall curve with AP."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = np.mean(y_true)
    
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(recall, precision, "b-", lw=2, label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=prevalence, color="k", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray,
    save_path: str, n_bins: int = 10, title: str = "Reliability Diagram",
):
    """Plot calibration reliability diagram.
    
    Perfect calibration = diagonal line.
    Below diagonal = overconfident.
    Above diagonal = underconfident.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_accs.append(y_true[mask].mean())
        bin_confs.append(y_prob[mask].mean())
        bin_counts.append(mask.sum())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), dpi=DPI,
                                     gridspec_kw={"height_ratios": [3, 1]})
    
    ax1.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color="steelblue",
            edgecolor="navy", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax1.set_ylabel("Fraction of Positives", fontsize=12)
    ax1.set_title(title, fontsize=14, weight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.05])

    ax2.bar(bin_confs, bin_counts, width=0.08, alpha=0.6, color="steelblue",
            edgecolor="navy")
    ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confidence_distribution(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    save_path: str,
):
    """Plot confidence distributions for correct vs incorrect predictions."""
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    correct_mask = y_true == y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

    axes[0].hist(confidence[correct_mask], bins=30, alpha=0.7, color="green",
                 label=f"Correct (n={correct_mask.sum()})", density=True)
    axes[0].hist(confidence[~correct_mask], bins=30, alpha=0.7, color="red",
                 label=f"Incorrect (n={(~correct_mask).sum()})", density=True)
    axes[0].set_xlabel("Prediction Confidence", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Confidence: Correct vs Incorrect", fontsize=13, weight="bold")
    axes[0].legend(fontsize=10)

    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    
    for mask, label, color in [(tp, "TP", "darkgreen"), (tn, "TN", "green"),
                                (fp, "FP", "orange"), (fn, "FN", "red")]:
        if mask.sum() > 0:
            axes[1].hist(confidence[mask], bins=20, alpha=0.5, label=f"{label} (n={mask.sum()})",
                        density=True)
    axes[1].set_xlabel("Prediction Confidence", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Confidence by Outcome Category", fontsize=13, weight="bold")
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_history(history: Dict, save_path: str, model_name: str = ""):
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    if "best_epoch" in history:
        ax1.axvline(x=history["best_epoch"], color="g", linestyle="--", alpha=0.5,
                    label=f"Best (epoch {history['best_epoch']})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} Loss", weight="bold")
    ax1.legend()
    
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} Accuracy", weight="bold")
    ax2.legend()
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(comparison_data: List[Dict], save_path: str):
    """Plot model comparison bar charts."""
    model_names = [d["model"] for d in comparison_data]
    
    metrics_to_plot = ["accuracy", "f1_score", "sensitivity", "specificity", "roc_auc", "pr_auc", "mcc"]
    available = [m for m in metrics_to_plot if m in comparison_data[0]]
    
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 5), dpi=DPI)
    if n_metrics == 1:
        axes = [axes]
    
    colors = sns.color_palette("deep", len(model_names))
    
    for i, metric in enumerate(available):
        values = [d.get(metric, 0) for d in comparison_data]
        bars = axes[i].bar(model_names, values, color=colors)
        axes[i].set_title(metric.replace("_", " ").title(), weight="bold", fontsize=11)
        axes[i].set_ylim([min(0.5, min(values) - 0.05), 1.02])
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", fontsize=9)
        axes[i].tick_params(axis="x", rotation=20)
    
    plt.suptitle("Model Comparison", fontsize=15, weight="bold", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
