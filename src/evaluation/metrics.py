"""
Evaluation metrics specifically tuned for medical image classification.
Computed wide range of metrics because accuracy alone is often misleading in clinical settings.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss, log_loss,
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    positive_label: int = 1,
) -> Dict:
    """Computed comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probability of positive class
        positive_label: Which class is considered positive (1 = SSA)
    
    Returns:
        Dictionary of all metrics with descriptions
    """

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prevalence = np.mean(y_true == positive_label)
    
    metrics = {}
    
    metrics["accuracy"] = {
        "value": float(accuracy_score(y_true, y_pred)),
        "description": "Overall correct predictions / total predictions",
        "medical_note": "MISLEADING with class imbalance. A model predicting all-HP gets 68.6% accuracy.",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["balanced_accuracy"] = {
        "value": float(balanced_accuracy_score(y_true, y_pred)),
        "description": "Average of sensitivity and specificity; corrects for class imbalance",
        "formula": "(Sensitivity + Specificity) / 2",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["sensitivity"] = {
        "value": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "description": "True Positive Rate — proportion of actual SSA correctly identified",
        "formula": "TP / (TP + FN)",
        "medical_note": "Critical in screening: missing SSA (precancerous) has clinical consequences",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["specificity"] = {
        "value": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "description": "True Negative Rate — proportion of actual HP correctly identified",
        "formula": "TN / (TN + FP)",
        "medical_note": "High specificity reduces unnecessary follow-up procedures",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["precision"] = {
        "value": float(precision_score(y_true, y_pred, zero_division=0)),
        "description": "Positive Predictive Value — of predicted SSA, how many are truly SSA",
        "formula": "TP / (TP + FP)",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["npv"] = {
        "value": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "description": "Negative Predictive Value — of predicted HP, how many are truly HP",
        "formula": "TN / (TN + FN)",
        "medical_note": "Low NPV means risk of missed precancerous lesions",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["f1_score"] = {
        "value": float(f1_score(y_true, y_pred, zero_division=0)),
        "description": "Harmonic mean of precision and recall",
        "formula": "2 * (Precision * Recall) / (Precision + Recall)",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["mcc"] = {
        "value": float(matthews_corrcoef(y_true, y_pred)),
        "description": "Matthews Correlation Coefficient — uses all 4 CM entries, robust to imbalance",
        "formula": "(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
        "range": "[-1, 1], 0 = random, 1 = perfect",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["roc_auc"] = {
        "value": float(roc_auc_score(y_true, y_prob)),
        "description": "Area under ROC curve — threshold-independent discrimination",
        "range": "[0.5, 1.0], 0.5 = random",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["pr_auc"] = {
        "value": float(average_precision_score(y_true, y_prob)),
        "description": "Area under Precision-Recall curve — better than ROC-AUC for imbalanced data",
        "medical_note": "More informative than ROC-AUC when positive class is minority (SSA = 31.4%)",
        "higher_is_better": True, "category": "primary",
    }
    
    metrics["cohens_kappa"] = {
        "value": float(cohen_kappa_score(y_true, y_pred)),
        "description": "Agreement beyond chance — comparable to inter-annotator agreement",
        "higher_is_better": True, "category": "secondary",
    }
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    for_ = fn / (fn + tn) if (fn + tn) > 0 else 0.0
    
    metrics["fpr"] = {
        "value": float(fpr),
        "description": "False Positive Rate = 1 - Specificity",
        "higher_is_better": False, "category": "secondary",
    }
    metrics["fnr"] = {
        "value": float(fnr),
        "description": "False Negative Rate = 1 - Sensitivity (missed SSA rate)",
        "medical_note": "Directly measures missed precancerous lesions",
        "higher_is_better": False, "category": "secondary",
    }
    metrics["fdr"] = {
        "value": float(fdr),
        "description": "False Discovery Rate = 1 - Precision",
        "higher_is_better": False, "category": "secondary",
    }
    metrics["for"] = {
        "value": float(for_),
        "description": "False Omission Rate = 1 - NPV",
        "higher_is_better": False, "category": "secondary",
    }

    sens = metrics["sensitivity"]["value"]
    spec = metrics["specificity"]["value"]
    
    metrics["lr_positive"] = {
        "value": float(sens / (1 - spec)) if spec < 1.0 else float("inf"),
        "description": "Likelihood Ratio + : how much more likely a positive result in SSA vs HP",
        "interpretation": ">10 = strong, 5-10 = moderate, 2-5 = weak",
        "higher_is_better": True, "category": "secondary",
    }
    
    metrics["lr_negative"] = {
        "value": float((1 - sens) / spec) if spec > 0 else float("inf"),
        "description": "Likelihood Ratio - : how much more likely a negative result in SSA vs HP",
        "interpretation": "<0.1 = strong, 0.1-0.2 = moderate",
        "higher_is_better": False, "category": "secondary",
    }
 
    if metrics["lr_negative"]["value"] > 0 and metrics["lr_negative"]["value"] != float("inf"):
        dor = metrics["lr_positive"]["value"] / metrics["lr_negative"]["value"]
    else:
        dor = float("inf")
    metrics["diagnostic_odds_ratio"] = {
        "value": float(dor) if dor != float("inf") else 999999.0,
        "description": "Overall diagnostic effectiveness",
        "higher_is_better": True, "category": "secondary",
    }

    metrics["brier_score"] = {
        "value": float(brier_score_loss(y_true, y_prob)),
        "description": "Mean squared difference between predicted probability and actual outcome",
        "range": "[0, 1], 0 = perfect calibration + discrimination",
        "medical_note": "Combines calibration and discrimination quality",
        "higher_is_better": False, "category": "primary",
    }
    
    metrics["log_loss"] = {
        "value": float(log_loss(y_true, y_prob)),
        "description": "Negative log-likelihood of predictions",
        "higher_is_better": False, "category": "secondary",
    }

    metrics["confusion_matrix"] = {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "category": "foundation",
    }
    
    metrics["prevalence"] = {
        "value": float(prevalence),
        "description": f"Proportion of positive class (SSA) in this set: {prevalence:.1%}",
        "category": "context",
    }
    
    metrics["macro_f1"] = {
        "value": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "description": "Unweighted average F1 across both classes",
        "higher_is_better": True, "category": "secondary",
    }
    
    return metrics


def compute_threshold_analysis(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[List[float]] = None
) -> List[Dict]:
    """
    Check how the model performs at various classification thresholds.
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        youdens_j = sens + spec - 1
        
        results.append({
            "threshold": t,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision": round(prec, 4),
            "npv": round(npv, 4),
            "f1": round(f1, 4),
            "youdens_j": round(youdens_j, 4),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
    
    return results


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    """
    Find the threshold that gives us the best balance of sensitivity and specificity.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youdens_j = tpr - fpr  # = sensitivity + specificity - 1
    optimal_idx = np.argmax(youdens_j)
    
    return {
        "optimal_threshold": float(thresholds[optimal_idx]),
        "youdens_j": float(youdens_j[optimal_idx]),
        "sensitivity_at_optimal": float(tpr[optimal_idx]),
        "specificity_at_optimal": float(1 - fpr[optimal_idx]),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence intervals to see how stable our metrics are.
    Note: I treated samples as independent here, which might be a slight oversimplification if multiple patches come from the same patient.
    """
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            score = metric_fn(y_true[idx], y_prob[idx])
            scores.append(score)
        except ValueError:
            continue
    
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(np.percentile(scores, alpha * 100)),
        "ci_upper": float(np.percentile(scores, (1 - alpha) * 100)),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def save_metrics(metrics: Dict, save_path: str) -> None:
    """Save metrics to JSON file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean = json.loads(json.dumps(metrics, default=convert))
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Metrics saved to {path}")
