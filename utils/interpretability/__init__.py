"""
Interpretability utilities for model explainability.

This module provides tools for visualizing and understanding model decisions,
including Grad-CAM for CNNs and attention visualization for MIL models.
"""

from .gradcam_3d import GradCAM3D, create_gradcam_for_resnet50_3d

__all__ = ['GradCAM3D', 'create_gradcam_for_resnet50_3d']

