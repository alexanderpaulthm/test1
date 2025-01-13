"""PyTorch hooks for feature extraction and gradient manipulation."""

from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor

class FeatureExtractor:
    """Hook to extract features from a layer."""

    def __init__(self, module: nn.Module):
        """Initialize the feature extractor.

        Args:
            module: Layer to extract features from
        """
        self.module = module
        self.features: Optional[Tensor] = None
        self.hook_handle = None

    def _hook_fn(self, module: nn.Module, input: tuple, output: Tensor):
        """Store the features during forward pass."""
        self.features = output.detach()

    def register(self):
        """Register the forward hook."""
        if self.hook_handle is None:
            self.hook_handle = self.module.register_forward_hook(self._hook_fn)

    def remove(self):
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class GradientExtractor:
    """Hook to extract gradients."""

    def __init__(self, module: nn.Module):
        """Initialize the gradient extractor.

        Args:
            module: Layer to extract gradients from
        """
        self.module = module
        self.gradients: Optional[Tensor] = None
        self.hook_handle = None

    def _hook_fn(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        """Store the gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def register(self):
        """Register the backward hook."""
        if self.hook_handle is None:
            self.hook_handle = self.module.register_backward_hook(self._hook_fn)

    def remove(self):
        """Remove the backward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

class LayerGradCam:
    """Hook combination for Grad-CAM computation."""

    def __init__(self, module: nn.Module):
        """Initialize Grad-CAM hooks.

        Args:
            module: Layer to analyze
        """
        self.module = module
        self.feature_extractor = FeatureExtractor(module)
        self.gradient_extractor = GradientExtractor(module)

    def register(self):
        """Register both hooks."""
        self.feature_extractor.register()
        self.gradient_extractor.register()

    def remove(self):
        """Remove both hooks."""
        self.feature_extractor.remove()
        self.gradient_extractor.remove()

    @property
    def features(self) -> Optional[Tensor]:
        """Get extracted features."""
        return self.feature_extractor.features

    @property
    def gradients(self) -> Optional[Tensor]:
        """Get extracted gradients."""
        return self.gradient_extractor.gradients