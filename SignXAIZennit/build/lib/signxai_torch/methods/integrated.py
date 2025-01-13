"""Integrated Gradients implementation."""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..core.attribution import GradientAttribution


class IntegratedGradients(GradientAttribution):
    """Integrated Gradients implementation."""

    def __init__(
            self,
            model: nn.Module,
            steps: int = 50,
            baseline: Optional[Tensor] = None
    ):
        super().__init__(model)
        self.steps = steps
        self.baseline = baseline

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute Integrated Gradients attribution."""
        # Create baseline if not provided
        if self.baseline is None:
            self.baseline = torch.zeros_like(inputs)
        elif self.baseline.shape != inputs.shape:
            raise ValueError(f"Baseline shape {self.baseline.shape} doesn't match input shape {inputs.shape}")

        # Validate target shape
        if target is not None:
            if isinstance(target, torch.Tensor):
                if target.dim() > 1:
                    raise ValueError(f"Target must be 1D tensor, got shape {target.shape}")
                if target.shape[0] != inputs.shape[0] and target.shape[0] != 1:
                    raise ValueError(f"Target size {target.shape[0]} doesn't match batch size {inputs.shape[0]}")

        # Calculate integration path
        alphas = torch.linspace(0, 1, self.steps, device=inputs.device)

        # Properly handle batch dimension for alphas
        if inputs.dim() == 4:  # Batched input
            alphas = alphas.view(-1, 1, 1, 1)
        else:  # Single input
            alphas = alphas.view(-1, 1, 1, 1)[:, :, :, :1]

        input_diff = inputs - self.baseline
        batch_size = inputs.size(0)

        # Initialize integrated gradients
        integrated_gradients = torch.zeros_like(inputs)

        # Compute gradients at each step
        for alpha in alphas:
            interpolated_input = self.baseline + alpha * input_diff
            interpolated_input.requires_grad_(True)

            # Forward pass
            output = self.model(interpolated_input)

            # Handle target
            if target is None:
                target = output.argmax(dim=1)
            elif isinstance(target, int):
                target = torch.full((batch_size,), target, device=inputs.device)
            elif isinstance(target, torch.Tensor) and target.numel() == 1:
                target = target.expand(batch_size)

            # Calculate gradients
            self.model.zero_grad()
            target_score = output[torch.arange(batch_size), target].sum()
            target_score.backward(retain_graph=True)

            if interpolated_input.grad is not None:
                integrated_gradients += interpolated_input.grad

        # Average gradients and multiply by input difference
        attribution = (inputs - self.baseline) * integrated_gradients / self.steps
        return attribution