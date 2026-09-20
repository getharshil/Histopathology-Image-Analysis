"""
Streamlit demo application for Histopathology Image Classification.

Run: streamlit run app/streamlit_app.py

DISCLAIMER: This is a research/educational prototype and NOT a clinical diagnostic system.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import get_eval_transforms, denormalize, IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet import ResNetModel
from src.models.efficientnet import EfficientNetModel
from src.models.baseline_cnn import BaselineCNN
from src.explainability.gradcam import GradCAM
from src.analysis.quantitative import analyze_activation_map


# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Histopathology AI Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CUSTOM CSS
# ======================================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        font-size: 0.85rem;
        color: #856404;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .hp-prediction { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .ssa-prediction { background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str):
    """Load trained model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = PROJECT_ROOT / "outputs" / "models"
    
    # Try finetuned first, then base
    for suffix in ["_finetuned_best.pth", "_best.pth"]:
        ckpt_path = models_dir / f"{model_name}{suffix}"
        if ckpt_path.exists():
            break
    else:
        return None, None
    
    if model_name == "baseline_cnn":
        model = BaselineCNN(num_classes=2)
    elif model_name == "resnet18":
        model = ResNetModel(num_classes=2, pretrained=False)
    elif model_name == "efficientnet_b0":
        model = EfficientNetModel(num_classes=2, pretrained=False)
    else:
        return None, None
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_image(model, device, image: Image.Image):
    """Run prediction on an image."""
    transform = get_eval_transforms(224)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
    
    prob_hp = probs[0, 0].item()
    prob_ssa = probs[0, 1].item()
    pred_class = "SSA" if prob_ssa > prob_hp else "HP"
    confidence = max(prob_hp, prob_ssa)
    
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "prob_hp": prob_hp,
        "prob_ssa": prob_ssa,
        "img_tensor": img_tensor.squeeze(0),
    }


def generate_gradcam(model, img_tensor, device):
    """Generate Grad-CAM heatmap."""
    target_layer = model.get_grad_cam_target_layer()
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_class, confidence = gradcam.generate(img_tensor.unsqueeze(0).to(device))
    overlay = gradcam.overlay_heatmap(img_tensor, heatmap, alpha=0.4)
    return heatmap, overlay


# ======================================================================
# MAIN APP
# ======================================================================
def main():
    st.markdown('<h1 class="main-header">🔬 Histopathology Image Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#666;">AI-powered classification of colorectal polyp images (HP vs SSA)</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>DISCLAIMER:</strong> This is a research/educational prototype and is NOT a clinical diagnostic system.
        It should not be used for medical decision-making. Always consult qualified healthcare professionals for diagnosis.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_name = st.selectbox(
            "Select Model",
            ["resnet18", "efficientnet_b0", "baseline_cnn"],
            index=0,
            help="Choose which trained model to use for prediction"
        )
        
        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        show_quantitative = st.checkbox("Show Quantitative Analysis", value=True)
        
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        **Task:** Binary classification  
        **Classes:** HP (Hyperplastic Polyp) vs SSA (Sessile Serrated Adenoma)  
        **Dataset:** MHIST (3,152 images)  
        **Models:** CNN, ResNet-18, EfficientNet-B0  
        """)
        
        # Show metrics if available
        metrics_path = PROJECT_ROOT / "outputs" / "metrics" / f"{model_name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.markdown("---")
            st.header("📈 Test Set Performance")
            for metric in ["roc_auc", "f1_score", "sensitivity", "specificity", "balanced_accuracy"]:
                if metric in metrics and isinstance(metrics[metric], dict):
                    val = metrics[metric].get("value", "N/A")
                    if isinstance(val, float):
                        st.metric(metric.replace("_", " ").title(), f"{val:.4f}")
    
    # Main content
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Upload a histopathology image (H&E stained, 224×224 recommended)",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Load model
        model, device = load_model(model_name)
        if model is None:
            st.error(f"❌ Model checkpoint not found for '{model_name}'. Run training first.")
            return
        
        # Predict
        with st.spinner("Running prediction..."):
            result = predict_image(model, device, image)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🖼️ Input Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Prediction")
            
            pred_class = result["predicted_class"]
            css_class = "hp-prediction" if pred_class == "HP" else "ssa-prediction"
            full_name = "Hyperplastic Polyp" if pred_class == "HP" else "Sessile Serrated Adenoma"
            
            st.markdown(f"""
            <div class="prediction-box {css_class}">
                {pred_class} — {full_name}
            </div>
            """, unsafe_allow_html=True)
            
            # Probability bars
            st.markdown("**Prediction Probabilities:**")
            st.progress(result["prob_hp"], text=f"HP (Hyperplastic): {result['prob_hp']:.1%}")
            st.progress(result["prob_ssa"], text=f"SSA (Sessile Serrated Adenoma): {result['prob_ssa']:.1%}")
            
            st.metric("Confidence", f"{result['confidence']:.1%}")
            
            if result["confidence"] < 0.7:
                st.warning("⚠️ Low confidence prediction — model is uncertain about this image.")
        
        # Grad-CAM
        if show_gradcam:
            st.markdown("---")
            st.subheader("🔥 Grad-CAM Explainability")
            st.caption("Heatmap shows regions the model focused on. This is model attention analysis, NOT lesion segmentation.")
            
            with st.spinner("Generating Grad-CAM..."):
                heatmap, overlay = generate_gradcam(model, result["img_tensor"], device)
            
            gcol1, gcol2, gcol3 = st.columns(3)
            
            with gcol1:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)
            
            with gcol2:
                st.markdown("**Heatmap**")
                heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                st.image(heatmap_colored, use_container_width=True)
            
            with gcol3:
                st.markdown("**Overlay**")
                st.image(overlay, use_container_width=True)
        
        # Quantitative analysis
        if show_quantitative:
            st.markdown("---")
            st.subheader("📐 Quantitative Activation Analysis")
            st.caption("Statistics derived from model attention maps — NOT ground-truth measurements.")
            
            if show_gradcam:
                stats = analyze_activation_map(heatmap)
                
                qcol1, qcol2, qcol3, qcol4 = st.columns(4)
                qcol1.metric("Activation Area", f"{stats['activation_percentage']:.1f}%")
                qcol2.metric("# Regions", str(stats['num_regions']))
                qcol3.metric("Mean Intensity", f"{stats['mean_activation_intensity']:.3f}")
                qcol4.metric("Largest Region", f"{stats['largest_region_area']} px")
                
                with st.expander("Full Quantitative Details"):
                    display_stats = {k: v for k, v in stats.items() if k != "bounding_boxes"}
                    st.json(display_stats)
    
    else:
        # Show example results from test set
        st.info("👆 Upload an image to get started, or explore the pre-computed results below.")
        
        # Show pre-computed figures if available
        gradcam_grid = PROJECT_ROOT / "outputs" / "gradcam"
        figures_dir = PROJECT_ROOT / "outputs" / "figures"
        
        if figures_dir.exists():
            st.markdown("---")
            st.subheader("📊 Pre-computed Results")
            
            tabs = st.tabs(["Model Comparison", "ROC Curves", "Grad-CAM", "Error Analysis"])
            
            with tabs[0]:
                comp_img = figures_dir / "model_comparison.png"
                if comp_img.exists():
                    st.image(str(comp_img), use_container_width=True)
                bench_img = figures_dir / "benchmark_comparison.png"
                if bench_img.exists():
                    st.image(str(bench_img), use_container_width=True)
            
            with tabs[1]:
                for m in ["resnet18", "efficientnet_b0", "baseline_cnn"]:
                    roc_img = figures_dir / f"{m}_roc_curve.png"
                    if roc_img.exists():
                        st.image(str(roc_img), caption=m, use_container_width=True)
            
            with tabs[2]:
                for m in ["resnet18", "efficientnet_b0"]:
                    gc_img = gradcam_grid / f"{m}_gradcam_grid.png"
                    if gc_img.exists():
                        st.image(str(gc_img), caption=f"{m} Grad-CAM", use_container_width=True)
            
            with tabs[3]:
                for m in ["resnet18", "efficientnet_b0"]:
                    for err_type in ["false_positives", "false_negatives", "high_confidence_errors"]:
                        err_img = PROJECT_ROOT / "outputs" / "errors" / f"{m}_{err_type}.png"
                        if err_img.exists():
                            st.image(str(err_img), use_container_width=True)


if __name__ == "__main__":
    main()
