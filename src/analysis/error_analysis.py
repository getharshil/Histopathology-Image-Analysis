"""Error analysis module for systematic failure investigation."""
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def find_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    filenames: List[str],
    annotator_counts: np.ndarray,
    label_names: List[str],
) -> Dict:
    """Categorize all predictions and identify error patterns.
    
    Investigates:
    - False positives and false negatives
    - High-confidence errors (most dangerous in medical AI)
    - Low-confidence correct predictions (uncertain but right)
    - Correlation between annotator disagreement and model errors
    """
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    correct = y_true == y_pred
    
    results = {
        "false_positives": [],
        "false_negatives": [],
        "high_confidence_errors": [],
        "low_confidence_correct": [],
    }
    
    for i in range(len(y_true)):
        entry = {
            "filename": filenames[i],
            "true_label": label_names[y_true[i]],
            "predicted_label": label_names[y_pred[i]],
            "confidence": round(float(confidence[i]), 4),
            "prob_ssa": round(float(y_prob[i]), 4),
            "annotator_ssa_count": int(annotator_counts[i]),
            "correct": bool(correct[i]),
        }
        
        if y_true[i] == 0 and y_pred[i] == 1: 
            results["false_positives"].append(entry)
        elif y_true[i] == 1 and y_pred[i] == 0: 
            results["false_negatives"].append(entry)
        
        if not correct[i] and confidence[i] > 0.9:
            results["high_confidence_errors"].append(entry)
        
        if correct[i] and confidence[i] < 0.6:
            results["low_confidence_correct"].append(entry)
    
    results["false_positives"].sort(key=lambda x: -x["confidence"])
    results["false_negatives"].sort(key=lambda x: -x["confidence"])
    results["high_confidence_errors"].sort(key=lambda x: -x["confidence"])
    

    results["annotator_agreement_analysis"] = analyze_agreement_vs_errors(
        y_true, y_pred, y_prob, annotator_counts
    )
    
    results["summary"] = {
        "total_errors": int((~correct).sum()),
        "total_correct": int(correct.sum()),
        "error_rate": round(float((~correct).mean()), 4),
        "n_false_positives": len(results["false_positives"]),
        "n_false_negatives": len(results["false_negatives"]),
        "n_high_confidence_errors": len(results["high_confidence_errors"]),
        "n_low_confidence_correct": len(results["low_confidence_correct"]),
    }
    
    return results


def analyze_agreement_vs_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    annotator_counts: np.ndarray,
) -> Dict:
    """Analyze relationship between annotator disagreement and model errors.
    
    Annotator counts: 0-7 (number of pathologists selecting SSA)
    - 0-1 or 6-7: High agreement (clear cases)
    - 2-5: Low agreement (ambiguous/difficult cases)
    """
    correct = y_true == y_pred
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    
    agreement_levels = {}
    for count in range(8):
        mask = annotator_counts == count
        if mask.sum() == 0:
            continue
        agreement_levels[f"annotators_{count}"] = {
            "count": int(mask.sum()),
            "accuracy": round(float(correct[mask].mean()), 4),
            "error_rate": round(float((~correct[mask]).mean()), 4),
            "mean_confidence": round(float(confidence[mask].mean()), 4),
        }
    
    high_agreement = (annotator_counts <= 1) | (annotator_counts >= 6)
    low_agreement = (annotator_counts >= 3) & (annotator_counts <= 4)
    
    return {
        "per_annotator_count": agreement_levels,
        "high_agreement_accuracy": round(float(correct[high_agreement].mean()), 4) if high_agreement.sum() > 0 else None,
        "low_agreement_accuracy": round(float(correct[low_agreement].mean()), 4) if low_agreement.sum() > 0 else None,
        "high_agreement_n": int(high_agreement.sum()),
        "low_agreement_n": int(low_agreement.sum()),
    }


def plot_error_examples(
    errors: List[Dict],
    image_dir: str,
    save_path: str,
    title: str = "Error Examples",
    max_images: int = 12,
):
    """Save grid of error examples with metadata."""
    errors = errors[:max_images]
    if not errors:
        print(f"No errors to plot for: {title}")
        return
    
    n = len(errors)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows), dpi=120)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for i, err in enumerate(errors):
        img_path = Path(image_dir) / err["filename"]
        try:
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
        except Exception:
            axes[i].text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                        transform=axes[i].transAxes)
        
        color = "red" if not err["correct"] else "green"
        axes[i].set_title(
            f"True: {err['true_label']} | Pred: {err['predicted_label']}\n"
            f"Conf: {err['confidence']:.2f} | Annot: {err['annotator_ssa_count']}/7",
            fontsize=9, color=color, weight="bold"
        )
        axes[i].axis("off")
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(title, fontsize=14, weight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_agreement_vs_accuracy(analysis: Dict, save_path: str):
    """Plot annotator agreement vs model accuracy."""
    per_count = analysis["per_annotator_count"]
    counts = []
    accuracies = []
    sizes = []
    
    for key in sorted(per_count.keys()):
        data = per_count[key]
        count_num = int(key.split("_")[1])
        counts.append(count_num)
        accuracies.append(data["accuracy"])
        sizes.append(data["count"])
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    color1 = "steelblue"
    bars = ax1.bar(counts, accuracies, color=color1, alpha=0.7, edgecolor="navy")
    ax1.set_xlabel("Number of Annotators Selecting SSA (out of 7)", fontsize=12)
    ax1.set_ylabel("Model Accuracy", fontsize=12, color=color1)
    ax1.set_ylim([0, 1.05])
    ax1.set_xticks(range(8))
    
    # Add count labels on bars
    for bar, acc, size in zip(bars, accuracies, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.2f}\n(n={size})", ha="center", fontsize=9)
    
    # Mark ambiguous zone
    ax1.axvspan(2.5, 4.5, alpha=0.1, color="red", label="Ambiguous zone")
    
    ax1.set_title("Model Accuracy vs Annotator Agreement\n"
                   "(Lower accuracy in ambiguous zone validates model difficulty correlation)",
                   fontsize=13, weight="bold")
    ax1.legend(loc="lower right")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
