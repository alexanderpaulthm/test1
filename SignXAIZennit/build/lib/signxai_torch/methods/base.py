"""Base gradient implementations."""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..core.attribution import GradientAttribution


class BaseGradient(GradientAttribution):
    """Base gradient implementation."""

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def _validate_target(self, target: Union[int, Tensor], outputs: Tensor) -> Tensor:
        """Validate and convert target to appropriate format."""
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]

        if isinstance(target, int):
            if target >= num_classes:
                raise ValueError(f"Target class {target} is invalid for model with {num_classes} classes")
            target = torch.full((batch_size,), target, device=outputs.device)
        elif isinstance(target, torch.Tensor):
            if target.dim() > 1:
                raise ValueError(f"Target must be 1D tensor, got shape {target.shape}")
            if target.shape[0] != batch_size and target.shape[0] != 1:
                raise ValueError(f"Target size {target.shape[0]} doesn't match batch size {batch_size}")
            if target.max() >= num_classes:
                raise ValueError(f"Target contains invalid class {target.max()} for model with {num_classes} classes")
            if target.shape[0] == 1:
                target = target.expand(batch_size)
        return target

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute base gradient attribution."""
        inputs = inputs.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        outputs = self.model(inputs)
        if target is None:
            target = outputs.argmax(dim=1)
        else:
            target = self._validate_target(target, outputs)

        # Create target vector
        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(outputs.shape[0]), target] = 1

        # Ensure output clone for view safety
        outputs = outputs.clone()

        # Compute gradients
        outputs.backward(gradient=one_hot)

        if inputs.grad is None:
            raise RuntimeError("No gradient computed. Check if model parameters require gradients.")

        return inputs.grad.clone()


class InputXGradient(BaseGradient):
    """Input times gradient implementation."""

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute input times gradient attribution."""
        gradients = super().attribute(inputs, target, **kwargs)
        return inputs.detach() * gradients.detach()


class GradientXSign(BaseGradient):
    """Gradient times SIGN implementation."""

    def __init__(
            self,
            model: nn.Module,
            mu: float = 0.0,
            epsilon: float = 1e-7
    ):
        super().__init__(model)
        self.mu = mu
        self.epsilon = epsilon

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute gradient times SIGN attribution."""
        gradients = super().attribute(inputs, target, **kwargs)
        inputs = inputs.detach()  # Detach inputs to prevent gradient flow

        # Create masks for positive and negative gradients
        pos_mask = gradients > self.epsilon
        neg_mask = gradients < -self.epsilon

        # Initialize output tensor
        result = torch.zeros_like(gradients)

        # Handle positive gradients
        if pos_mask.any():
            result[pos_mask] = torch.abs(inputs[pos_mask] * gradients[pos_mask]) * (1 + self.mu)

        # Handle negative gradients
        if neg_mask.any():
            result[neg_mask] = -torch.abs(inputs[neg_mask] * gradients[neg_mask]) * (1 - self.mu)

        return result.clone()