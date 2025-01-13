from typing import Optional, Union, Dict
import torch
import torch.nn as nn
from torch import Tensor

from .base import LRP


class EpsilonLRP(LRP):
    """Epsilon-LRP implementation."""

    def __init__(
            self,
            model: nn.Module,
            epsilon: float = 1e-7,
            use_sign: bool = False,  # Add this parameter
            mu: float = 0.0,        # Add this parameter
            layer_rules: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, layer_rules)
        self.epsilon = epsilon
        self.use_sign = use_sign  # Store the parameter
        self.mu = mu             # Store the parameter

    def _compute_relevance(
            self,
            module: nn.Module,
            input: Tensor,
            output: Tensor,
            relevance: Tensor,
            rule: str
    ) -> Tensor:
        """Compute layer relevance using epsilon rule."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Get weights and handle bias
            weights = module.weight
            if module.bias is not None:
                bias = module.bias.view(-1, 1, 1) if isinstance(module, nn.Conv2d) else module.bias
            else:
                bias = torch.zeros(weights.size(0), device=weights.device)

            # For Conv2d, reshape bias to match output dimensions
            if isinstance(module, nn.Conv2d):
                bias = bias.view(1, -1, 1, 1)

            # Calculate forward pass activations
            z = output  # Already stored activations
            s = torch.where(z > 0, z, torch.zeros_like(z)) + self.epsilon

            # Calculate relevance for positive and negative contributions
            relevance_pos = torch.where(z > 0, relevance, torch.zeros_like(relevance))
            relevance_neg = torch.where(z < 0, relevance, torch.zeros_like(relevance))

            if isinstance(module, nn.Conv2d):
                # Compute backward relevance for convolutional layer
                rel_pos = torch.nn.functional.conv_transpose2d(
                    relevance_pos / s,
                    weights.clamp(min=0),
                    padding=module.padding,
                    stride=module.stride
                )
                rel_neg = torch.nn.functional.conv_transpose2d(
                    relevance_neg / s,
                    weights.clamp(max=0),
                    padding=module.padding,
                    stride=module.stride
                )
            else:  # Linear layer
                # Compute backward relevance for fully connected layer
                rel_pos = torch.matmul(relevance_pos / s, weights.clamp(min=0).t())
                rel_neg = torch.matmul(relevance_neg / s, weights.clamp(max=0).t())

            return (rel_pos + rel_neg) * input

        elif isinstance(module, nn.BatchNorm2d):
            # For BatchNorm, we redistribute the relevance proportionally
            norm = output.abs().sum(dim=1, keepdim=True) + self.epsilon
            scaled_relevance = relevance * (input.abs() / norm)
            return scaled_relevance

        # For other layers (like ReLU), just pass through the relevance
        return relevance

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute Epsilon-LRP attribution."""
        inputs.requires_grad_(True)
        self._relevances.clear()

        # Forward pass
        outputs = self.model(inputs)

        # Handle target selection
        if target is None:
            target = outputs.argmax(dim=1)
        else:
            target = self._validate_target(target, outputs)

        # Initialize relevance at output layer
        batch_size = outputs.size(0)
        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(batch_size), target] = 1.0

        # Backward pass with relevance propagation
        outputs.backward(gradient=one_hot)

        # Apply epsilon rule to input gradients
        input_grad = inputs.grad.detach()
        stabilizer = torch.sign(input_grad) * self.epsilon
        attribution = input_grad * inputs / (input_grad + stabilizer)

        return attribution