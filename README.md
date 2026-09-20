# Explainable AI for Histopathology Image Analysis

An end-to-end deep learning pipeline for classifying colorectal polyp images (Hyperplastic Polyp vs. Sessile Serrated Adenoma) using the MHIST dataset. Designed to demonstrate medical AI engineering, including comprehensive clinical evaluation, model calibration, Grad-CAM explainability, and quantitative activation analysis.

---

## Project Overview

This project builds an automated classification system for histopathology patches, targeting the distinction between Hyperplastic Polyps (HP) and precancerous Sessile Serrated Adenomas (SSA).

**Key Features:**
- **Clinical Evaluation:** Beyond accuracy, we report Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV), and MCC to properly contextualize performance in a medical setting.
- **Model Calibration:** Reliability diagrams and Expected Calibration Error (ECE) analysis to ensure the model's confidence scores are trustworthy.
- **Explainable AI (Grad-CAM):** Visualizes the regions of the tissue patch that most influenced the model's prediction.
- **Quantitative Activation Analysis:** Extracts region-level statistics (area, count, density) from attention maps, moving beyond qualitative heatmaps.
- **Error Analysis & Annotator Agreement:** Explores the correlation between model errors and human pathologist disagreement, highlighting ambiguous cases.
- **Performance Engineering:** Benchmarks inference latency (CPU vs GPU vs AMP) and model size trade-offs across different architectures.

---

## Dataset: MHIST
The [MHIST dataset](https://bmirds.github.io/MHIST/) contains 3,152 fixed-size (224×224) H&E-stained images.
- **Classes:** HP (Benign, 68.6%) vs. SSA (Precancerous, 31.4%)
- **Annotations:** Ground truth is based on the majority vote of 7 board-certified gastrointestinal pathologists.
- **Splits:** We use the official test set (977 images) for final evaluation, and split the official train set into train (85%) and validation (15%).

---

## Getting Started

### 1. Requirements & Installation
Install the necessary dependencies using Python 3.10+:
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
The master script executes the entire project: EDA, training 3 models (Baseline CNN, ResNet-18, EfficientNet-B0), full metric evaluation, Grad-CAM generation, and hardware benchmarking.
```bash
python scripts/run_pipeline.py
```
*Note: This will take approximately 10-20 minutes depending on your GPU.*

All results, checkpoints, and visualizations will be saved to the `outputs/` directory.

### 3. Run the Streamlit Demo App
Launch the interactive web application to upload images, view predictions, and explore Grad-CAM heatmaps interactively:
```bash
streamlit run app/streamlit_app.py
```

---

## Methodology

1. **Preprocessing & Augmentation:** Uses ImageNet normalization and biologically plausible augmentations (flips, rotations, color jitter) to handle stain variation and arbitrary tissue orientation.
2. **Architectures:** 
   - `BaselineCNN`: A custom 4-block CNN built from scratch.
   - `ResNet-18`: Transfer learning baseline (matches the original MHIST paper).
   - `EfficientNet-B0`: Evaluated for high parameter efficiency.
3. **Training Strategy:** Two-phase transfer learning. We first freeze the pretrained backbone to train the classification head, then unfreeze later layers for fine-tuning using a lower learning rate.
4. **Evaluation:** Computes a comprehensive suite of 20+ metrics, including ROC/PR curves, threshold optimization (Youden's J), and bootstrap confidence intervals.

---

## Repository Structure

```
.
├── app/
│   └── streamlit_app.py        # Interactive web demo
├── configs/
│   ├── baseline.yaml           # Hyperparameters for CNN
│   ├── resnet18.yaml           # Hyperparameters for ResNet-18
│   └── efficientnet.yaml       # Hyperparameters for EfficientNet-B0
├── scripts/
│   └── run_pipeline.py         # Master execution script
├── src/
│   ├── analysis/               # Error & quantitative activation analysis
│   ├── data/                   # Dataset loader & preprocessing
│   ├── evaluation/             # Metrics, calibration, & plotting
│   ├── explainability/         # Grad-CAM implementation
│   ├── models/                 # Neural network architectures
│   └── training/               # Unified training loop & mixed precision
├── outputs/                    # Generated during execution (models, figures, metrics)
├── requirements.txt
└── README.md
```

---

## Limitations & Ethical Considerations
- **Patch-Level Analysis:** This pipeline operates on 224×224 cropped patches. A full digital pathology system would require a Whole Slide Image (WSI) preprocessing pipeline (tiling, tissue detection).
- **Not a Segmentation Tool:** Grad-CAM visualizations represent model attention, not exact ground-truth lesion boundaries.
- **Clinical Readiness:** This is an educational/research prototype. Do not use for actual medical diagnosis.

---

## Resume Bullets Example
- Designed and implemented an end-to-end explainable deep learning pipeline in PyTorch for histopathology image classification (HP vs SSA), utilizing transfer learning with ResNet-18 and EfficientNet-B0.
- Engineered a comprehensive medical evaluation suite tracking 20+ metrics (Sensitivity, Specificity, ROC-AUC, PR-AUC), coupled with model calibration analysis (ECE) and threshold optimization.
- Developed a Grad-CAM explainability module and extracted quantitative region-level activation statistics to interpret model attention patterns.
- Correlated model errors with human annotator disagreement levels, demonstrating strong scientific rigor in error analysis.
- Benchmarked inference latency across CPU and GPU (FP32/AMP), and deployed the final optimized model via a Streamlit web application.
