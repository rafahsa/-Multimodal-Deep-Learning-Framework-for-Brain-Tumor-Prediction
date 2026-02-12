"""
3D Grad-CAM Implementation for ResNet50-3D

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for 3D CNN models, specifically designed for ResNet50-3D brain tumor classification.

Grad-CAM generates heatmaps that highlight the spatial regions in the input volume
that are most important for the model's prediction.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class GradCAM3D:
    """
    3D Grad-CAM implementation for visualizing CNN decisions.
    
    Computes Grad-CAM heatmaps by:
    1. Capturing activations from a target layer (e.g., layer4)
    2. Computing gradients of the target class logit w.r.t. activations
    3. Computing channel-wise weights (mean of gradients)
    4. Generating weighted activation map
    5. Applying ReLU and normalizing to [0, 1]
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model (must be in eval mode)
            target_layer: Target layer to extract activations from (e.g., model.layer4)
            device: Device to run computation on
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        
        def forward_hook(module, input, output):
            """Store activations from forward pass."""
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Store gradients from backward pass."""
            # grad_output is a tuple, take first element
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for given input.
        
        Args:
            input_tensor: Input volume of shape (1, C, D, H, W)
            target_class: Target class index (0=LGG, 1=HGG). If None, uses predicted class.
            retain_graph: Whether to retain computation graph (default: False)
        
        Returns:
            Grad-CAM heatmap as numpy array of shape (D, H, W), values in [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Clear previous activations/gradients
        self.activations = None
        self.gradients = None
        
        # Move input to device
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)
        
        # Determine target class
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        
        # Get target class score
        target_score = logits[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=retain_graph)
        
        # Check that we have activations and gradients
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check that forward hook is registered correctly.")
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check that backward hook is registered correctly.")
        
        # Get shapes
        # activations: (1, C, D', H', W')
        # gradients: (1, C, D', H', W')
        B, C, D, H, W = self.activations.shape
        
        # Compute channel-wise weights: mean of gradients over spatial dimensions
        # alpha_c = mean_{d,h,w} (dY/dA_c)
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        
        # Compute weighted activation map
        # CAM = sum_c (alpha_c * A_c)
        cam = (weights * self.activations).sum(dim=1, keepdim=False)  # (1, D', H', W')
        cam = cam.squeeze(0)  # (D', H', W')
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # All zeros or constant
            cam = torch.zeros_like(cam)
        
        # Convert to numpy
        cam_np = cam.detach().cpu().numpy()
        
        # Upsample to input resolution if needed
        input_shape = input_tensor.shape[2:]  # (D, H, W)
        cam_shape = cam_np.shape  # (D', H', W')
        
        if cam_shape != input_shape:
            # Upsample using trilinear interpolation
            cam_tensor = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0)  # (1, 1, D', H', W')
            cam_upsampled = F.interpolate(
                cam_tensor,
                size=input_shape,
                mode='trilinear',
                align_corners=False
            )
            cam_np = cam_upsampled.squeeze(0).squeeze(0).numpy()  # (D, H, W)
        
        return cam_np
    
    def generate_cam_with_prediction(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate Grad-CAM heatmap and return prediction info.
        
        Args:
            input_tensor: Input volume of shape (1, C, D, H, W)
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            Tuple of (cam_heatmap, prediction_info)
            - cam_heatmap: numpy array of shape (D, H, W), values in [0, 1]
            - prediction_info: dict with 'predicted_class', 'probabilities', 'target_class'
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Move input to device
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction first (without gradients)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            predicted_class = int(torch.argmax(logits, dim=1).item())
            probabilities = {
                'LGG': float(probs[0, 0].item()),
                'HGG': float(probs[0, 1].item())
            }
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class=target_class)
        
        # Determine actual target class used
        if target_class is None:
            target_class = predicted_class
        
        prediction_info = {
            'predicted_class': predicted_class,
            'predicted_class_name': 'HGG' if predicted_class == 1 else 'LGG',
            'probabilities': probabilities,
            'target_class': target_class,
            'target_class_name': 'HGG' if target_class == 1 else 'LGG'
        }
        
        return cam, prediction_info


def create_gradcam_for_resnet50_3d(
    model: nn.Module,
    device: torch.device
) -> GradCAM3D:
    """
    Create Grad-CAM instance for ResNet50-3D model.
    
    Args:
        model: ResNet50-3D model (ResNet3D instance from create_resnet50_3d)
        device: Device to run computation on
    
    Returns:
        GradCAM3D instance with hooks registered on layer4
    """
    # Access the underlying ResNet3D model
    # create_resnet50_3d returns ResNet3D directly, but check for wrapper
    if hasattr(model, 'model'):
        # Model is wrapped (e.g., ResNet50_3D wrapper)
        resnet_model = model.model
    else:
        # Model is ResNet3D directly
        resnet_model = model
    
    # Target layer: layer4 (last convolutional block before global pooling)
    if not hasattr(resnet_model, 'layer4'):
        raise ValueError(f"Model does not have layer4 attribute. Model type: {type(resnet_model)}")
    
    target_layer = resnet_model.layer4
    
    logger.info(f"Grad-CAM target layer: layer4")
    logger.info(f"Target layer output shape (approximate): (1, 2048, 8, 8, 8) for 128³ input")
    
    return GradCAM3D(model=model, target_layer=target_layer, device=device)

