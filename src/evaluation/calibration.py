"""
Model calibration analysis.
"""
import numpy as np
from typing import Dict, Tuple


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Calculated the Expected Calibration Error (ECE).
    This measures the average gap between how confident the model is and how accurate it actually is.
    We usually use 10 bins, though this is somewhat arbitrary. Lower ECE is better.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi)
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        
        count = mask.sum()
        if count == 0:
            bin_data.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "count": 0, "avg_confidence": 0.0, "avg_accuracy": 0.0, "gap": 0.0,
            })
            continue
        
        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        gap = abs(avg_accuracy - avg_confidence)
        ece += (count / len(y_true)) * gap
        
        bin_data.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "count": int(count),
            "avg_confidence": round(avg_confidence, 4),
            "avg_accuracy": round(avg_accuracy, 4),
            "gap": round(gap, 4),
        })
    
    return float(ece), {"bins": bin_data, "n_bins": n_bins}


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Get the Maximum Calibration Error (the worst-case gap across all bins)."""
    _, info = compute_ece(y_true, y_prob, n_bins)
    gaps = [b["gap"] for b in info["bins"] if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def compute_confidence_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict:
    """
    Break down prediction confidence by outcome (TP, TN, FP, FN).
    This is for if we wanted to know if the model is highly confident when it makes a mistake, which is dangerous.
    """
    # Confidence = max(p, 1-p) for the predicted class
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    def stats(mask):
        vals = confidence[mask]
        if len(vals) == 0:
            return {"count": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
        return {
            "count": int(mask.sum()),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
        }
    
    return {
        "overall": stats(np.ones(len(y_true), dtype=bool)),
        "correct": stats((y_true == y_pred)),
        "incorrect": stats((y_true != y_pred)),
        "tp": stats(tp_mask),
        "tn": stats(tn_mask),
        "fp": stats(fp_mask),
        "fn": stats(fn_mask),
        "high_confidence_errors": int(((y_true != y_pred) & (confidence > 0.9)).sum()),
        "low_confidence_correct": int(((y_true == y_pred) & (confidence < 0.6)).sum()),
    }


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Scale logits by a temperature parameter.
    T > 1 softens probabilities (good for overconfident models).
    T < 1 sharpens them.
    """
    scaled = logits / temperature
    # Softmax
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def find_optimal_temperature(
    y_true: np.ndarray, logits: np.ndarray, lr: float = 0.01, max_iter: int = 100
) -> float:
    """
    Find the best temperature to calibrate the model by minimizing the negative log-likelihood.
    """
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss
    
    def nll(t):
        probs = temperature_scale(logits, t)
        return log_loss(y_true, probs[:, 1])
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)
