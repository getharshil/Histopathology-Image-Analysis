"""
Grad-CAM implementation for visualizing what the model is looking at.

Note that this wil show us the model's attention (which regions contributed most to the prediction), rather than an exact segmentation of the lesion. Since we're constrained by the feature map resolution (e.g., 7x7 for ResNet-18), it's a bit coarse, but it's a great sanity check to ensure the model isn't cheating by looking at background artifacts.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional

from src.data.preprocessing import denormalize


class GradCAM:
    """Grad-CAM visualization for CNN models."""
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, H, W)
            target_class: Class to generate CAM for (None = predicted class)
        
        Returns:
            heatmap: Normalized heatmap (H, W) in [0, 1]
            pred_class: Predicted class index
            confidence: Prediction confidence (softmax probability)
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
            
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, confidence
    
    def overlay_heatmap(
        self,
        image_tensor: torch.Tensor,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay Grad-CAM heatmap on original image.
        
        Args:
            image_tensor: Original preprocessed tensor (3, H, W)
            heatmap: Grad-CAM heatmap (H, W) in [0, 1]
            alpha: Overlay transparency
        
        Returns:
            overlay: RGB image with heatmap overlay (H, W, 3) in [0, 255]
        """
        img = denormalize(image_tensor).permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = (heatmap_colored * alpha + img * (1 - alpha)).astype(np.uint8)
        
        return overlay
    
    def create_visualization(
        self,
        image_tensor: torch.Tensor,
        label_names: list,
        true_label: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Create full Grad-CAM visualization with metadata.
        
        Returns:
            combined: Side-by-side image (original | heatmap | overlay)
            info: Dictionary with prediction details
        """
        heatmap, pred_class, confidence = self.generate(image_tensor)
        
        original = denormalize(image_tensor).permute(1, 2, 0).numpy()
        original = (original * 255).astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = self.overlay_heatmap(image_tensor, heatmap)
        
        combined = np.concatenate([original, heatmap_colored, overlay], axis=1)
        
        info = {
            "predicted_class": label_names[pred_class],
            "confidence": confidence,
            "true_label": label_names[true_label] if true_label is not None else None,
            "correct": (pred_class == true_label) if true_label is not None else None,
            "heatmap": heatmap,
        }
        
        return combined, info
