"""Implementation of the SIGN method using PyTorch."""

from typing import Optional, Union, List

import torch
import torch.nn as nn
from torch import Tensor
from zennit.core import Stabilizer

from ..core.attribution import GradientAttribution


class SIGN(GradientAttribution):
    """SIGN (SIGNed explanations) method implementation."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1e-7,
        mu: float = 0.0
    ):
        """Initialize SIGN method.

        Args:
            model: PyTorch model to analyze
            epsilon: Small constant for numerical stability
            mu: Mu parameter for SIGN method
        """
        super().__init__(model)
        self.epsilon = epsilon
        self.mu = mu
        self.stabilizer = Stabilizer(epsilon=epsilon)

    def _stabilize(self, tensor: Tensor) -> Tensor:
        """Apply stabilization to tensor.

        Args:
            tensor: Input tensor

        Returns:
            Stabilized tensor
        """
        return self.stabilizer(tensor)

    def attribute(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        **kwargs
    ) -> Tensor:
        """Compute SIGN attributions.

        Args:
            inputs: Input tensor to analyze
            target: Target class or tensor
            **kwargs: Additional parameters

        Returns:
            Attribution scores
        """
        if inputs.numel() == 0:
            raise ValueError("Empty input tensor")

        inputs.requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        batch_size = outputs.size(0)

        # Handle target selection
        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)
            target = target.expand(batch_size)
        elif isinstance(target, torch.Tensor) and target.numel() == 1:
            target = target.expand(batch_size)

        # Compute gradients with proper batch handling
        target_scores = outputs[torch.arange(batch_size), target]
        grad_tensor = torch.ones_like(target_scores)
        gradients = torch.autograd.grad(target_scores, inputs, grad_outputs=grad_tensor,
                                      create_graph=True)[0]

        # Apply SIGN method
        pos_mask = gradients > 0
        neg_mask = gradients < 0

        # Apply stabilization and SIGN formulation
        relevance = torch.zeros_like(inputs)

        if pos_mask.any():
            pos_grad = self._stabilize(gradients[pos_mask])
            relevance[pos_mask] = inputs[pos_mask] * ((1 + self.mu) * pos_grad)

        if neg_mask.any():
            neg_grad = self._stabilize(-gradients[neg_mask])
            relevance[neg_mask] = -inputs[neg_mask] * ((1 - self.mu) * neg_grad)

        # Normalize attribution per sample
        relevance_flat = relevance.view(batch_size, -1)
        max_abs = relevance_flat.abs().max(dim=1, keepdim=True)[0].view(batch_size, 1, 1, 1)
        max_abs = torch.where(max_abs > self.epsilon, max_abs, torch.ones_like(max_abs))
        relevance = relevance / max_abs

        return relevance

    def get_integrated_gradients(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        steps: int = 50,
        baseline: Optional[Tensor] = None
    ) -> Tensor:
        """Compute integrated SIGN attributions.

        Args:
            inputs: Input tensor
            target: Target class or tensor
            steps: Number of integration steps
            baseline: Optional baseline tensor (zeros if None)

        Returns:
            Integrated SIGN attribution
        """
        if baseline is None:
            baseline = torch.zeros_like(inputs)

        path = torch.linspace(0, 1, steps=steps, device=inputs.device)
        integrated_grad = torch.zeros_like(inputs)

        for alpha in path:
            interpolated = baseline + alpha * (inputs - baseline)
            attribution = self.attribute(interpolated, target)
            integrated_grad += attribution / steps

        return integrated_grad