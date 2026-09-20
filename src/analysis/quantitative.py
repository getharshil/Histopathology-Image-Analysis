"""Quantitative activation analysis from Grad-CAM heatmaps.

These metrics describe activation patterns, not biological structures.
"""
import cv2
import numpy as np
from typing import Dict, List


def analyze_activation_map(heatmap: np.ndarray, threshold: float = 0.5) -> Dict:
    """Extract quantitative statistics from a Grad-CAM heatmap.
    
    This converts qualitative attention visualization into measurable quantities,
    demonstrating image quantification skills relevant to AIRA Matrix's work.
    
    Args:
        heatmap: Normalized Grad-CAM heatmap (H, W) in [0, 1]
        threshold: Activation threshold for region detection
    
    Returns:
        Dictionary of quantitative activation statistics
    """
    h, w = heatmap.shape
    total_pixels = h * w
    
    binary_mask = (heatmap >= threshold).astype(np.uint8)
    activated_pixels = binary_mask.sum()
    
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    num_regions = num_components - 1
    
    region_areas = []
    bounding_boxes = []
    
    for i in range(1, num_components):
        area = stats[i, cv2.CC_STAT_AREA]
        region_areas.append(area)
        bounding_boxes.append({
            "x": int(stats[i, cv2.CC_STAT_LEFT]),
            "y": int(stats[i, cv2.CC_STAT_TOP]),
            "width": int(stats[i, cv2.CC_STAT_WIDTH]),
            "height": int(stats[i, cv2.CC_STAT_HEIGHT]),
        })

    activated_values = heatmap[binary_mask == 1] if activated_pixels > 0 else np.array([0])
    
    results = {

        "activated_pixels": int(activated_pixels),
        "total_pixels": int(total_pixels),
        "activation_percentage": round(float(activated_pixels / total_pixels * 100), 2),

        "num_regions": int(num_regions),
        "largest_region_area": int(max(region_areas)) if region_areas else 0,
        "smallest_region_area": int(min(region_areas)) if region_areas else 0,
        "mean_region_area": round(float(np.mean(region_areas)), 1) if region_areas else 0,
        "median_region_area": round(float(np.median(region_areas)), 1) if region_areas else 0,

        "mean_activation_intensity": round(float(activated_values.mean()), 4),
        "max_activation_intensity": round(float(heatmap.max()), 4),
        "activation_density": round(float(heatmap.mean()), 4),
 
        "bounding_boxes": bounding_boxes,
        "threshold_used": threshold,
    }
    

    if num_regions > 0:
        largest_idx = np.argmax(region_areas) + 1 
        results["largest_region_centroid"] = {
            "x": round(float(centroids[largest_idx][0]), 1),
            "y": round(float(centroids[largest_idx][1]), 1),
        }
    
    return results


def compare_class_activations(
    hp_activations: List[Dict],
    ssa_activations: List[Dict],
) -> Dict:
    """Compare activation statistics between HP and SSA predictions.
    
    This can reveal whether the model uses different spatial patterns for each class,
    which could indicate biologically meaningful attention differences.
    """
    def aggregate(activations: List[Dict]) -> Dict:
        if not activations:
            return {}
        return {
            "mean_activation_percentage": round(np.mean([a["activation_percentage"] for a in activations]), 2),
            "std_activation_percentage": round(np.std([a["activation_percentage"] for a in activations]), 2),
            "mean_num_regions": round(np.mean([a["num_regions"] for a in activations]), 2),
            "mean_activation_density": round(np.mean([a["activation_density"] for a in activations]), 4),
            "mean_largest_region": round(np.mean([a["largest_region_area"] for a in activations]), 1),
            "n_samples": len(activations),
        }
    
    return {
        "HP_activations": aggregate(hp_activations),
        "SSA_activations": aggregate(ssa_activations),
        "note": "These statistics describe model attention patterns, not ground-truth lesion morphology.",
    }
