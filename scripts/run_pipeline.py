"""
Master pipeline: Runs the entire project end-to-end.

Usage:
    python scripts/run_pipeline.py

This script executes all phases:
1 Data validation & EDA
2 Train Baseline CNN, ResNet-18, EfficientNet-B0
3 Evaluate all models (full metrics suite)
4 Generate Grad-CAM visualizations
5 Error analysis + annotator agreement
6 Quantitative activation analysis
7 GPU benchmarking
8 Model comparison
"""
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.seed import set_seed, get_device
from src.utils.config import load_config, save_config
from src.data.dataset import MHISTDataset
from src.data.preprocessing import get_train_transforms, get_eval_transforms, denormalize
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet import ResNetModel
from src.models.efficientnet import EfficientNetModel
from src.training.trainer import Trainer
from src.evaluation.metrics import (
    compute_all_metrics, compute_threshold_analysis,
    find_optimal_threshold, compute_bootstrap_ci, save_metrics,
)
from src.evaluation.calibration import (
    compute_ece, compute_mce, compute_confidence_analysis,
)
from src.evaluation.plots import (
    plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
    plot_reliability_diagram, plot_confidence_distribution,
    plot_training_history, plot_model_comparison,
)
from src.explainability.gradcam import GradCAM
from src.analysis.error_analysis import (
    find_errors, plot_error_examples, plot_agreement_vs_accuracy,
)
from src.analysis.quantitative import analyze_activation_map, compare_class_activations

# ======================================================================
# CONFIGURATION
# ======================================================================
ANNOTATIONS_CSV = str(PROJECT_ROOT / "annotations.csv")
IMAGE_DIR = str(PROJECT_ROOT / "images")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
SEED = 42

# Output subdirs
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")
ERRORS_DIR = os.path.join(OUTPUT_DIR, "errors")

for d in [FIGURES_DIR, METRICS_DIR, MODELS_DIR, GRADCAM_DIR, ERRORS_DIR]:
    os.makedirs(d, exist_ok=True)


def print_header(text: str):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


# ======================================================================
# PHASE 1: DATA VALIDATION & EDA
# ======================================================================
def run_eda():
    print_header("PHASE 1: DATA VALIDATION & EXPLORATORY ANALYSIS")
    
    df = pd.read_csv(ANNOTATIONS_CSV)
    print(f"\nDataset size: {len(df)} images")
    print(f"Columns: {list(df.columns)}")
    
    # Class distribution
    print("\n--- Class Distribution ---")
    class_counts = df["Majority Vote Label"].value_counts()
    print(class_counts)
    print(f"Class ratio (HP:SSA): {class_counts['HP'] / class_counts['SSA']:.2f}:1")
    print(f"HP prevalence: {class_counts['HP'] / len(df) * 100:.1f}%")
    print(f"SSA prevalence: {class_counts['SSA'] / len(df) * 100:.1f}%")
    
    # Partition distribution
    print("\n--- Partition Distribution ---")
    for part in ["train", "test"]:
        part_df = df[df["Partition"] == part]
        part_counts = part_df["Majority Vote Label"].value_counts()
        print(f"  {part}: {len(part_df)} images (HP: {part_counts.get('HP', 0)}, SSA: {part_counts.get('SSA', 0)})")
    
    # Annotator agreement
    print("\n--- Annotator Agreement Distribution ---")
    ann_col = "Number of Annotators who Selected SSA (Out of 7)"
    ann_counts = df[ann_col].value_counts().sort_index()
    for count, n in ann_counts.items():
        print(f"  {count}/7 annotators selected SSA: {n} images ({n/len(df)*100:.1f}%)")
    
    ambiguous = df[(df[ann_col] >= 3) & (df[ann_col] <= 4)]
    print(f"\n  Ambiguous cases (3-4/7): {len(ambiguous)} ({len(ambiguous)/len(df)*100:.1f}%)")
    
    # Validate images
    print("\n--- Image Validation ---")
    missing = 0
    corrupted = 0
    sizes = set()
    for _, row in df.iterrows():
        img_path = Path(IMAGE_DIR) / row["Image Name"]
        if not img_path.exists():
            missing += 1
            continue
        try:
            img = Image.open(img_path)
            sizes.add(img.size)
        except Exception:
            corrupted += 1
    
    print(f"  Missing: {missing}")
    print(f"  Corrupted: {corrupted}")
    print(f"  Image sizes found: {sizes}")
    
    # Pixel statistics (sample 200 images)
    print("\n--- Pixel Statistics (sampled) ---")
    sample_df = df.sample(min(200, len(df)), random_state=SEED)
    pixel_means = []
    pixel_stds = []
    for _, row in sample_df.iterrows():
        img = np.array(Image.open(Path(IMAGE_DIR) / row["Image Name"]))
        pixel_means.append(img.mean(axis=(0, 1)))
        pixel_stds.append(img.std(axis=(0, 1)))
    
    pixel_means = np.array(pixel_means)
    pixel_stds = np.array(pixel_stds)
    print(f"  Mean RGB: [{pixel_means[:, 0].mean():.1f}, {pixel_means[:, 1].mean():.1f}, {pixel_means[:, 2].mean():.1f}]")
    print(f"  Std RGB:  [{pixel_stds[:, 0].mean():.1f}, {pixel_stds[:, 1].mean():.1f}, {pixel_stds[:, 2].mean():.1f}]")
    
    # --- EDA Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Class distribution
    colors = ["#4C72B0", "#DD8452"]
    axes[0, 0].bar(class_counts.index, class_counts.values, color=colors, edgecolor="black")
    for i, (cls, cnt) in enumerate(class_counts.items()):
        axes[0, 0].text(i, cnt + 20, f"{cnt}\n({cnt/len(df)*100:.1f}%)", ha="center", fontsize=11, weight="bold")
    axes[0, 0].set_title("Class Distribution", fontsize=13, weight="bold")
    axes[0, 0].set_ylabel("Count")
    
    # Annotator agreement
    axes[0, 1].bar(ann_counts.index, ann_counts.values, color="steelblue", edgecolor="navy")
    axes[0, 1].axvspan(2.5, 4.5, alpha=0.15, color="red", label="Ambiguous zone")
    axes[0, 1].set_title("Annotator Agreement Distribution", fontsize=13, weight="bold")
    axes[0, 1].set_xlabel("# Annotators selecting SSA (out of 7)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend()
    
    # RGB channel histograms
    for ch, color, name in [(0, "red", "R"), (1, "green", "G"), (2, "blue", "B")]:
        axes[1, 0].hist(pixel_means[:, ch], bins=30, alpha=0.5, color=color, label=name)
    axes[1, 0].set_title("Mean Pixel Intensity Distribution (per image)", fontsize=13, weight="bold")
    axes[1, 0].set_xlabel("Mean Intensity")
    axes[1, 0].legend()
    
    # Sample images
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Sample Images (see montage below)", fontsize=13, weight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "eda_overview.png"), bbox_inches="tight")
    plt.close()
    
    # Sample montage
    fig, axes = plt.subplots(2, 8, figsize=(20, 6), dpi=120)
    for row_idx, label in enumerate(["HP", "SSA"]):
        samples = df[df["Majority Vote Label"] == label].sample(8, random_state=SEED)
        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            img = Image.open(Path(IMAGE_DIR) / sample["Image Name"])
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")
            ann = sample["Number of Annotators who Selected SSA (Out of 7)"]
            axes[row_idx, col_idx].set_title(f"SSA:{ann}/7", fontsize=8)
        axes[row_idx, 0].set_ylabel(label, fontsize=14, weight="bold", rotation=0, labelpad=30)
    
    plt.suptitle("Sample Images: HP (top) vs SSA (bottom)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sample_montage.png"), bbox_inches="tight")
    plt.close()
    
    print("\n[DONE] EDA complete. Figures saved to outputs/figures/")
    return df


# ======================================================================
# PHASE 2-3: TRAINING
# ======================================================================
def create_dataloaders(config: dict):
    """Create train/val/test dataloaders from config."""
    train_transform = get_train_transforms(
        config["data"]["image_size"],
        config["data"].get("augmentation", "moderate"),
    )
    eval_transform = get_eval_transforms(config["data"]["image_size"])
    
    train_ds = MHISTDataset(ANNOTATIONS_CSV, IMAGE_DIR, split="train",
                            transform=train_transform, val_ratio=config["data"]["val_ratio"],
                            seed=config.get("seed", 42))
    val_ds = MHISTDataset(ANNOTATIONS_CSV, IMAGE_DIR, split="val",
                          transform=eval_transform, val_ratio=config["data"]["val_ratio"],
                          seed=config.get("seed", 42))
    test_ds = MHISTDataset(ANNOTATIONS_CSV, IMAGE_DIR, split="test",
                           transform=eval_transform, seed=config.get("seed", 42))
    
    common_kwargs = dict(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ds)} | {train_ds.get_class_counts()}")
    print(f"  Val:   {len(val_ds)} | {val_ds.get_class_counts()}")
    print(f"  Test:  {len(test_ds)} | {test_ds.get_class_counts()}")
    
    return train_loader, val_loader, test_loader, train_ds


def build_model(config: dict, device: torch.device):
    """Build model from config."""
    model_name = config["model"]["name"]
    
    if model_name == "baseline_cnn":
        model = BaselineCNN(
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "resnet18":
        model = ResNetModel(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            freeze_backbone=config["model"]["freeze_backbone"],
            dropout=config["model"]["dropout"],
        )
    elif model_name == "efficientnet_b0":
        model = EfficientNetModel(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            freeze_backbone=config["model"]["freeze_backbone"],
            dropout=config["model"]["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def train_model(config_path: str, device: torch.device) -> dict:
    """Train a single model from a config file."""
    config = load_config(config_path)
    model_name = config["model"]["name"]
    print_header(f"TRAINING: {model_name.upper()}")
    
    set_seed(config.get("seed", 42))
    
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(config)
    model = build_model(config, device)
    
    criterion = nn.CrossEntropyLoss()
    if config["training"].get("use_class_weights", False):
        weights = torch.tensor(train_ds.get_class_weights(), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"Using class weights: {weights.tolist()}")
    
    opt_name = config["training"].get("optimizer", "adam").lower()
    lr = config["training"]["learning_rate"]
    wd = config["training"].get("weight_decay", 0)
    
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=config["training"].get("scheduler_patience", 5),
        factor=config["training"].get("scheduler_factor", 0.5),
    )
    
    # Train Phase 1 (frozen backbone if applicable)
    trainer = Trainer(
        model=model, device=device, optimizer=optimizer, criterion=criterion,
        scheduler=scheduler,
        use_amp=config["training"].get("use_amp", True),
        output_dir=OUTPUT_DIR, model_name=model_name,
        patience=config["training"].get("patience", 10),
    )
    
    history = trainer.fit(train_loader, val_loader, num_epochs=config["training"]["epochs"])
    
    # Phase 2: Fine-tune (for transfer learning models)
    finetune_epochs = config["training"].get("finetune_epochs", 0)
    if finetune_epochs > 0 and hasattr(model, "unfreeze_backbone"):
        print_header(f"FINE-TUNING: {model_name.upper()}")
        
        if model_name == "resnet18":
            model.unfreeze_backbone(config["training"].get("unfreeze_from", "layer3"))
        elif model_name == "efficientnet_b0":
            model.unfreeze_backbone(config["training"].get("unfreeze_from_block", 5))
        
        ft_lr = config["training"].get("finetune_lr", 1e-4)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=ft_lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        
        trainer_ft = Trainer(
            model=model, device=device, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, use_amp=config["training"].get("use_amp", True),
            output_dir=OUTPUT_DIR, model_name=f"{model_name}_finetuned",
            patience=config["training"].get("patience", 8),
        )
        
        ft_history = trainer_ft.fit(train_loader, val_loader, num_epochs=finetune_epochs)
        
        # Merge histories
        for key in ["train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"]:
            history[key].extend(ft_history[key])
        history["total_train_time"] = history.get("total_train_time", 0) + ft_history.get("total_train_time", 0)
        history["finetune_best_epoch"] = ft_history.get("best_epoch", 0)
    
    plot_training_history(history, os.path.join(FIGURES_DIR, f"{model_name}_training.png"), model_name)
    
    return history


# ======================================================================
# PHASE 4: EVALUATION
# ======================================================================
@torch.no_grad()
def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Run inference and collect predictions."""
    model.eval()
    all_labels = []
    all_probs = []
    all_filenames = []
    all_annotator_counts = []
    all_logits = []
    
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(SSA)
        all_logits.extend(outputs.cpu().numpy())
        all_filenames.extend(batch["filename"])
        all_annotator_counts.extend(batch["annotator_ssa_count"].numpy())
    
    return {
        "y_true": np.array(all_labels),
        "y_prob": np.array(all_probs),
        "y_pred": (np.array(all_probs) >= 0.5).astype(int),
        "logits": np.array(all_logits),
        "filenames": all_filenames,
        "annotator_counts": np.array(all_annotator_counts),
    }


def full_evaluation(model_name: str, model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Run comprehensive evaluation for a model."""
    print_header(f"EVALUATION: {model_name.upper()}")
    
    preds = evaluate_model(model, test_loader, device)
    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]
    
    # All metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    
    # Threshold analysis
    threshold_analysis = compute_threshold_analysis(y_true, y_prob)
    metrics["threshold_analysis"] = threshold_analysis
    
    # Optimal threshold
    optimal = find_optimal_threshold(y_true, y_prob)
    metrics["optimal_threshold"] = optimal
    
    # Calibration
    ece, ece_info = compute_ece(y_true, y_prob)
    mce = compute_mce(y_true, y_prob)
    metrics["ece"] = {"value": ece, "description": "Expected Calibration Error", "higher_is_better": False}
    metrics["mce"] = {"value": mce, "description": "Maximum Calibration Error", "higher_is_better": False}
    
    # Confidence analysis
    conf_analysis = compute_confidence_analysis(y_true, y_pred, y_prob)
    metrics["confidence_analysis"] = conf_analysis
    
    # Bootstrap CIs for key metrics
    print("  Computing bootstrap CIs...")
    for metric_name, metric_fn in [
        ("roc_auc_ci", roc_auc_score),
        ("pr_auc_ci", average_precision_score),
    ]:
        ci = compute_bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=500)
        metrics[metric_name] = ci
    
    # Save metrics
    save_metrics(metrics, os.path.join(METRICS_DIR, f"{model_name}_metrics.json"))
    
    # Print summary
    print(f"\n--- {model_name} Test Results ---")
    primary = {k: v for k, v in metrics.items() if isinstance(v, dict) and v.get("category") == "primary"}
    for name, data in primary.items():
        val = data["value"]
        direction = "(higher is better)" if data.get("higher_is_better", True) else "(lower is better)"
        print(f"  {name:25s}: {val:.4f} {direction}")
    
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
    
    # Plots
    labels = MHISTDataset.LABEL_NAMES
    plot_confusion_matrix(y_true, y_pred, labels,
                          os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix.png"),
                          f"{model_name} Confusion Matrix")
    plot_roc_curve(y_true, y_prob,
                   os.path.join(FIGURES_DIR, f"{model_name}_roc_curve.png"),
                   f"{model_name} ROC Curve")
    plot_pr_curve(y_true, y_prob,
                  os.path.join(FIGURES_DIR, f"{model_name}_pr_curve.png"),
                  f"{model_name} Precision-Recall Curve")
    plot_reliability_diagram(y_true, y_prob,
                             os.path.join(FIGURES_DIR, f"{model_name}_reliability.png"),
                             title=f"{model_name} Reliability Diagram")
    plot_confidence_distribution(y_true, y_pred, y_prob,
                                 os.path.join(FIGURES_DIR, f"{model_name}_confidence.png"))
    
    # Threshold analysis table
    print(f"\n--- Threshold Analysis ---")
    print(f"  {'Threshold':>9} {'Sens':>7} {'Spec':>7} {'Prec':>7} {'NPV':>7} {'F1':>7} {'J':>7}")
    for t in threshold_analysis:
        print(f"  {t['threshold']:>9.2f} {t['sensitivity']:>7.4f} {t['specificity']:>7.4f} "
              f"{t['precision']:>7.4f} {t['npv']:>7.4f} {t['f1']:>7.4f} {t['youdens_j']:>7.4f}")
    print(f"\n  Optimal threshold (Youden's J): {optimal['optimal_threshold']:.3f} "
          f"(Sens={optimal['sensitivity_at_optimal']:.4f}, Spec={optimal['specificity_at_optimal']:.4f})")
    
    return metrics, preds


# ======================================================================
# PHASE 5: GRAD-CAM & QUANTITATIVE ANALYSIS
# ======================================================================
def run_gradcam_analysis(model_name: str, model: nn.Module, test_loader: DataLoader, device: torch.device, preds: dict):
    """Generate Grad-CAM visualizations and quantitative analysis."""
    print_header(f"GRAD-CAM & QUANTITATIVE ANALYSIS: {model_name.upper()}")
    
    target_layer = model.get_grad_cam_target_layer()
    gradcam = GradCAM(model, target_layer)
    
    y_true, y_pred, y_prob = preds["y_true"], preds["y_pred"], preds["y_prob"]
    filenames = preds["filenames"]
    confidence = np.where(y_pred == 1, y_prob, 1 - y_prob)
    
    # Categorize indices
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    
    eval_transform = get_eval_transforms(224)
    label_names = MHISTDataset.LABEL_NAMES
    
    # Select examples for each category
    categories = {
        "true_positive": tp_idx,
        "true_negative": tn_idx,
        "false_positive": fp_idx,
        "false_negative": fn_idx,
    }
    
    # Quantitative activation stats
    hp_activations = []
    ssa_activations = []
    
    all_examples = []
    
    for cat_name, indices in categories.items():
        if len(indices) == 0:
            continue
        
        # Pick top-3 by confidence
        sorted_by_conf = indices[np.argsort(-confidence[indices])][:3]
        
        for idx in sorted_by_conf:
            img_path = Path(IMAGE_DIR) / filenames[idx]
            img = Image.open(img_path).convert("RGB")
            img_tensor = eval_transform(img)
            
            combined, info = gradcam.create_visualization(
                img_tensor, label_names, true_label=int(y_true[idx])
            )
            
            # Save individual visualization
            save_name = f"{model_name}_{cat_name}_{filenames[idx]}"
            plt.figure(figsize=(15, 5), dpi=120)
            plt.imshow(combined)
            plt.title(
                f"{cat_name.upper()} | True: {label_names[y_true[idx]]} | "
                f"Pred: {info['predicted_class']} | Conf: {info['confidence']:.3f}",
                fontsize=12, weight="bold"
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(GRADCAM_DIR, save_name), bbox_inches="tight")
            plt.close()
            
            # Quantitative analysis
            activation_stats = analyze_activation_map(info["heatmap"])
            activation_stats["filename"] = filenames[idx]
            activation_stats["category"] = cat_name
            activation_stats["true_label"] = label_names[y_true[idx]]
            activation_stats["predicted_label"] = info["predicted_class"]
            
            if y_pred[idx] == 0:
                hp_activations.append(activation_stats)
            else:
                ssa_activations.append(activation_stats)
            
            all_examples.append(activation_stats)
    
    # Run quantitative analysis on more samples
    print("\n  Running quantitative analysis on all test predictions...")
    for idx in range(min(len(y_true), 200)):
        img_path = Path(IMAGE_DIR) / filenames[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = eval_transform(img)
            heatmap, _, _ = gradcam.generate(img_tensor)
            stats = analyze_activation_map(heatmap)
            if y_pred[idx] == 0:
                hp_activations.append(stats)
            else:
                ssa_activations.append(stats)
        except Exception as e:
            print(f"  Warning: Failed for {filenames[idx]}: {e}")
    
    # Class comparison
    comparison = compare_class_activations(hp_activations, ssa_activations)
    
    # Save quantitative results
    quant_results = {
        "examples": all_examples,
        "class_comparison": comparison,
    }
    with open(os.path.join(METRICS_DIR, f"{model_name}_quantitative.json"), "w") as f:
        json.dump(quant_results, f, indent=2, default=str)
    
    print(f"\n--- Quantitative Activation Comparison ---")
    for cls in ["HP_activations", "SSA_activations"]:
        data = comparison.get(cls, {})
        if data:
            print(f"  {cls}:")
            print(f"    Mean activation %: {data.get('mean_activation_percentage', 'N/A')}")
            print(f"    Mean regions: {data.get('mean_num_regions', 'N/A')}")
            print(f"    Mean density: {data.get('mean_activation_density', 'N/A')}")
    
    # Grad-CAM grid
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), dpi=120)
    example_idx = 0
    for row, (cat_name, indices) in enumerate(categories.items()):
        if len(indices) == 0:
            for col in range(3):
                axes[row, col].axis("off")
            continue
        sorted_idx = indices[np.argsort(-confidence[indices])][:3]
        for col, idx in enumerate(sorted_idx):
            img_path = Path(IMAGE_DIR) / filenames[idx]
            img = Image.open(img_path).convert("RGB")
            img_tensor = eval_transform(img)
            combined, info = gradcam.create_visualization(img_tensor, label_names, int(y_true[idx]))
            axes[row, col].imshow(combined)
            axes[row, col].set_title(
                f"True:{label_names[y_true[idx]]} Pred:{info['predicted_class']} "
                f"({info['confidence']:.2f})", fontsize=9, weight="bold"
            )
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(cat_name.replace("_", " ").title(), fontsize=12,
                                 weight="bold", rotation=90, labelpad=10)
    
    plt.suptitle(f"{model_name} Grad-CAM: Original | Heatmap | Overlay", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(GRADCAM_DIR, f"{model_name}_gradcam_grid.png"), bbox_inches="tight")
    plt.close()
    
    print(f"[DONE] Grad-CAM visualizations saved to {GRADCAM_DIR}")
    return comparison


# ======================================================================
# PHASE 6: ERROR ANALYSIS
# ======================================================================
def run_error_analysis(model_name: str, preds: dict):
    """Run systematic error analysis."""
    print_header(f"ERROR ANALYSIS: {model_name.upper()}")
    
    errors = find_errors(
        preds["y_true"], preds["y_pred"], preds["y_prob"],
        preds["filenames"], preds["annotator_counts"],
        MHISTDataset.LABEL_NAMES,
    )
    
    # Save
    with open(os.path.join(METRICS_DIR, f"{model_name}_errors.json"), "w") as f:
        json.dump(errors, f, indent=2)
    
    print(f"\n--- Error Summary ---")
    for k, v in errors["summary"].items():
        print(f"  {k}: {v}")
    
    # Plot error examples
    plot_error_examples(errors["false_positives"], IMAGE_DIR,
                        os.path.join(ERRORS_DIR, f"{model_name}_false_positives.png"),
                        f"{model_name}: False Positives (predicted SSA, actually HP)")
    plot_error_examples(errors["false_negatives"], IMAGE_DIR,
                        os.path.join(ERRORS_DIR, f"{model_name}_false_negatives.png"),
                        f"{model_name}: False Negatives (predicted HP, actually SSA)")
    plot_error_examples(errors["high_confidence_errors"], IMAGE_DIR,
                        os.path.join(ERRORS_DIR, f"{model_name}_high_confidence_errors.png"),
                        f"{model_name}: High-Confidence Errors (>90% confidence)")
    
    # Annotator agreement vs accuracy plot
    if errors["annotator_agreement_analysis"]["per_annotator_count"]:
        plot_agreement_vs_accuracy(
            errors["annotator_agreement_analysis"],
            os.path.join(FIGURES_DIR, f"{model_name}_agreement_vs_accuracy.png"),
        )
    
    return errors


# ======================================================================
# PHASE 7: GPU BENCHMARKING
# ======================================================================
def benchmark_model(model_name: str, model: nn.Module, device: torch.device, image_size: int = 224):
    """Benchmark inference speed: CPU vs GPU vs AMP."""
    print_header(f"BENCHMARKING: {model_name.upper()}")
    
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    n_runs = 50
    warmup = 10
    
    results = {}
    
    # CPU benchmark
    model_cpu = model.cpu()
    dummy_cpu = dummy.cpu()
    
    for _ in range(warmup):
        with torch.no_grad():
            model_cpu(dummy_cpu)
    
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model_cpu(dummy_cpu)
    cpu_time = (time.time() - start) / n_runs * 1000  # ms
    results["cpu_ms_per_image"] = round(cpu_time, 2)
    results["cpu_images_per_sec"] = round(1000 / cpu_time, 1)
    
    # GPU benchmark
    if device.type == "cuda":
        model_gpu = model.to(device)
        dummy_gpu = dummy.to(device)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / n_runs * 1000
        results["gpu_ms_per_image"] = round(gpu_time, 2)
        results["gpu_images_per_sec"] = round(1000 / gpu_time, 1)
        results["speedup_gpu_vs_cpu"] = round(cpu_time / gpu_time, 2)
        
        # AMP benchmark
        for _ in range(warmup):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad(), torch.amp.autocast("cuda"):
                model_gpu(dummy_gpu)
        torch.cuda.synchronize()
        amp_time = (time.time() - start) / n_runs * 1000
        results["amp_ms_per_image"] = round(amp_time, 2)
        results["amp_images_per_sec"] = round(1000 / amp_time, 1)
        results["speedup_amp_vs_cpu"] = round(cpu_time / amp_time, 2)
        
        # GPU memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model_gpu(dummy_gpu)
        results["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    
    # Model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    results["model_size_mb"] = round(param_size / 1024**2, 2)
    results["total_params"] = sum(p.numel() for p in model.parameters())
    results["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n--- Benchmark Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Save
    with open(os.path.join(METRICS_DIR, f"{model_name}_benchmark.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


# ======================================================================
# MAIN PIPELINE
# ======================================================================
def main():
    start_time = time.time()
    
    print_header("HISTOPATHOLOGY AI PIPELINE")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    
    set_seed(SEED)
    device = get_device()
    
    # Phase 1: EDA
    df = run_eda()
    
    # Phase 2-3: Train all models
    configs = [
        str(PROJECT_ROOT / "configs" / "baseline.yaml"),
        str(PROJECT_ROOT / "configs" / "resnet18.yaml"),
        str(PROJECT_ROOT / "configs" / "efficientnet.yaml"),
    ]
    
    all_histories = {}
    all_metrics = {}
    all_preds = {}
    all_benchmarks = {}
    
    for config_path in configs:
        config = load_config(config_path)
        model_name = config["model"]["name"]
        
        # Try to load the finetuned checkpoint first, then the base
        ft_path = Path(MODELS_DIR) / f"{model_name}_finetuned_best.pth"
        base_path = Path(MODELS_DIR) / f"{model_name}_best.pth"
        
        # Determine if we should train or skip
        ckpt_exists = ft_path.exists() or base_path.exists()
        
        if ckpt_exists:
            print(f"\nSkipping training for {model_name} because checkpoint exists.")
            history = {"total_train_time": 0}  # Dummy history
            all_histories[model_name] = history
        else:
            # Train
            history = train_model(config_path, device)
            all_histories[model_name] = history
        
        # Load best model for evaluation
        model = build_model(config, device)
        
        ckpt_path = ft_path if ft_path.exists() else base_path
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint: {ckpt_path.name}")
        
        # Create test loader
        eval_transform = get_eval_transforms(config["data"]["image_size"])
        test_ds = MHISTDataset(ANNOTATIONS_CSV, IMAGE_DIR, split="test", transform=eval_transform)
        test_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"],
                                 num_workers=config["data"]["num_workers"], pin_memory=True)
        
        # Phase 4: Evaluate
        metrics, preds = full_evaluation(model_name, model, test_loader, device)
        all_metrics[model_name] = metrics
        all_preds[model_name] = preds
        
        # Phase 5: Grad-CAM (only for best/last model, save time)
        run_gradcam_analysis(model_name, model, test_loader, device, preds)
        
        # Phase 6: Error analysis
        run_error_analysis(model_name, preds)
        
        # Phase 7: Benchmark
        benchmarks = benchmark_model(model_name, model, device)
        all_benchmarks[model_name] = benchmarks
    
    # ======================================================================
    # MODEL COMPARISON
    # ======================================================================
    print_header("MODEL COMPARISON")
    
    comparison_data = []
    for model_name in all_metrics:
        m = all_metrics[model_name]
        b = all_benchmarks[model_name]
        h = all_histories[model_name]
        
        row = {
            "model": model_name,
            "accuracy": m["accuracy"]["value"],
            "balanced_accuracy": m["balanced_accuracy"]["value"],
            "sensitivity": m["sensitivity"]["value"],
            "specificity": m["specificity"]["value"],
            "precision": m["precision"]["value"],
            "f1_score": m["f1_score"]["value"],
            "mcc": m["mcc"]["value"],
            "roc_auc": m["roc_auc"]["value"],
            "pr_auc": m["pr_auc"]["value"],
            "brier_score": m["brier_score"]["value"],
            "ece": m["ece"]["value"],
            "total_params": b["total_params"],
            "model_size_mb": b["model_size_mb"],
            "cpu_ms": b["cpu_ms_per_image"],
            "gpu_ms": b.get("gpu_ms_per_image", "N/A"),
            "gpu_memory_mb": b.get("gpu_memory_mb", "N/A"),
            "training_time_s": round(h.get("total_train_time", 0), 1),
        }
        comparison_data.append(row)
    
    # Print comparison table
    print(f"\n{'Model':<18} {'Acc':>6} {'Bal.Acc':>8} {'Sens':>6} {'Spec':>6} {'F1':>6} {'MCC':>6} "
          f"{'ROC-AUC':>8} {'PR-AUC':>8} {'Params':>10} {'GPU ms':>7}")
    print("-" * 120)
    for r in comparison_data:
        print(f"{r['model']:<18} {r['accuracy']:>6.4f} {r['balanced_accuracy']:>8.4f} "
              f"{r['sensitivity']:>6.4f} {r['specificity']:>6.4f} {r['f1_score']:>6.4f} {r['mcc']:>6.4f} "
              f"{r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['total_params']:>10,} {str(r['gpu_ms']):>7}")
    
    # Save comparison
    with open(os.path.join(METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    plot_model_comparison(comparison_data, os.path.join(FIGURES_DIR, "model_comparison.png"))
    
    # ======================================================================
    # BENCHMARK COMPARISON PLOT
    # ======================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    model_names = [d["model"] for d in comparison_data]
    
    # Latency
    cpu_times = [d["cpu_ms"] for d in comparison_data]
    gpu_times = [d["gpu_ms"] if isinstance(d["gpu_ms"], (int, float)) else 0 for d in comparison_data]
    x = np.arange(len(model_names))
    axes[0].bar(x - 0.2, cpu_times, 0.4, label="CPU", color="steelblue")
    axes[0].bar(x + 0.2, gpu_times, 0.4, label="GPU", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=15)
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Inference Latency", weight="bold")
    axes[0].legend()
    
    # Model size
    sizes = [d["model_size_mb"] for d in comparison_data]
    axes[1].bar(model_names, sizes, color="mediumpurple", edgecolor="indigo")
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Model Size", weight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    
    # Accuracy vs params scatter
    for d in comparison_data:
        axes[2].scatter(d["total_params"], d["roc_auc"], s=100, zorder=5)
        axes[2].annotate(d["model"], (d["total_params"], d["roc_auc"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[2].set_xlabel("Parameters")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].set_title("Accuracy vs Complexity", weight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "benchmark_comparison.png"), bbox_inches="tight")
    plt.close()
    
    total_time = time.time() - start_time
    print_header(f"PIPELINE COMPLETE - Total time: {total_time/60:.1f} minutes")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Metrics: {METRICS_DIR}")
    print(f"  Models:  {MODELS_DIR}")
    print(f"  Grad-CAM: {GRADCAM_DIR}")
    print(f"  Errors: {ERRORS_DIR}")


if __name__ == "__main__":
    main()
