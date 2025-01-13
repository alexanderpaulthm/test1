"""GradCAM implementation using PyTorch."""

from typing import Optional, Union, List, Type, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..core.attribution import GradientAttribution
from ..modules.hooks import FeatureExtractor

class GradCAM(GradientAttribution):
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_relu: bool = True
    ):
        """Initialize GradCAM.

        Args:
            model: PyTorch model to analyze
            target_layer: Layer to extract features from. If None, tries to find the last convolutional layer
            use_relu: Whether to apply ReLU to the final result
        """
        super().__init__(model)

        if target_layer is None:
            target_layer = self._find_last_conv(model)
            if target_layer is None:
                raise ValueError("Could not find a convolutional layer. Please specify target_layer.")

        self.target_layer = target_layer
        self.use_relu = use_relu
        self.features = None
        self.gradients = None

    def _find_last_conv(self, model: nn.Module) -> Optional[nn.Module]:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                last_conv = module
        return last_conv

    def _save_gradient(self, grad):
        """Save gradients for GradCAM computation."""
        self.gradients = grad

    def _save_features(self, module, input, output):
        """Save features for GradCAM computation."""
        self.features = output
        output.requires_grad_(True)
        output.register_hook(self._save_gradient)

    def attribute(self, inputs: Tensor, target: Optional[Union[int, Tensor]] = None,
                  interpolate_size: Optional[tuple] = None, **kwargs) -> Tensor:
        """Compute GradCAM attribution.

        Args:
            inputs: Input tensor to analyze
            target: Target class or tensor
            interpolate_size: Optional size to interpolate attribution to
            **kwargs: Additional arguments

        Returns:
            GradCAM attribution maps
        """
        # Register hooks
        handle = self.target_layer.register_forward_hook(self._save_features)

        # Ensure input requires gradients
        inputs.requires_grad_(True)

        # Forward pass
        outputs = self.model(inputs)
        batch_size = outputs.size(0)

        # Handle target selection
        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)
            target = target.expand(batch_size)

        # Create one-hot encoding of targets
        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(batch_size), target] = 1.0

        # Compute gradients
        outputs.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and features
        gradients = self.gradients
        features = self.features

        # Remove hook
        handle.remove()

        if gradients is None:
            raise ValueError("No gradients found. Make sure target_layer produces gradients.")

        # Calculate weights by global average pooling the gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Generate heatmap
        cam = torch.sum(weights * features, dim=1)

        if self.use_relu:
            cam = torch.relu(cam)

        # Normalize each sample
        cam_flat = cam.view(batch_size, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-7)

        if interpolate_size is not None:
            cam = nn.functional.interpolate(
                cam.unsqueeze(1),
                size=interpolate_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

        return cam

    def attribute_with_integrated_gradients(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        steps: int = 50,
        interpolate_size: Optional[tuple] = None
    ) -> Tensor:
        """Compute Grad-CAM with integrated gradients.

        Args:
            inputs: Input tensor
            target: Target class or tensor
            steps: Number of steps for integration
            interpolate_size: Optional size to interpolate to

        Returns:
            Integrated Grad-CAM attribution
        """
        if target is not None and not isinstance(target, torch.Tensor):
            target = torch.tensor([target], device=inputs.device)

        baseline = torch.zeros_like(inputs)
        integrated_cam = torch.zeros((inputs.size(0), *inputs.shape[2:]), device=inputs.device)

        for step in range(steps):
            alpha = step / (steps - 1)
            interpolated = baseline + alpha * (inputs - baseline)
            cam = self.attribute(interpolated, target=target, interpolate_size=interpolate_size)
            integrated_cam += cam

        integrated_cam /= steps
        return integrated_cam