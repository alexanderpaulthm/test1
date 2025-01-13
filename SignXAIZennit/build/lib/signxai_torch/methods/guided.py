"""Guided Backpropagation implementation using PyTorch."""

from typing import Optional, Union, Type

import torch
import torch.nn as nn
from torch import Tensor

from ..core.attribution import GradientAttribution


class GuidedBackprop(GradientAttribution):
    """Guided Backpropagation implementation."""

    def __init__(self, model: nn.Module):
        """Initialize Guided Backpropagation."""
        super().__init__(model)
        self._hooks = []
        self._replace_relu()

    class _GuidedReLU(torch.autograd.Function):
        """Custom ReLU function for guided backpropagation."""

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.relu(input)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = torch.zeros_like(grad_output)
            # Mask based on both input and gradient values
            pos_mask = (input > 0) & (grad_output > 0)
            grad_input[pos_mask] = grad_output[pos_mask] * 0.5
            return grad_input

    def _replace_relu(self):
        """Replace all ReLU activations with guided version."""
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(self._forward_hook)
                self._hooks.append(hook)

    def _forward_hook(self, module: nn.Module, input: tuple, output: Tensor):
        """Forward hook to replace ReLU's output with guided version."""
        return self._GuidedReLU.apply(input[0])

    def attribute(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        **kwargs
    ) -> Tensor:
        """Compute Guided Backpropagation attribution."""
        inputs.requires_grad_(True)
        self.model.zero_grad()

        outputs = self.model(inputs)

        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)
            target = target.expand(outputs.size(0))

        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(len(outputs)), target] = 1.0
        outputs.backward(gradient=one_hot)

        gradients = inputs.grad.clone()
        # Ensure zero gradients for small values
        gradients[torch.abs(gradients) < 1e-6] = 0.0
        return gradients


class DeconvNet(GradientAttribution):
    """DeconvNet implementation."""

    def __init__(self, model: nn.Module):
        """Initialize DeconvNet."""
        super().__init__(model)
        self._hooks = []
        self._replace_relu()

    class _DeconvReLU(torch.autograd.Function):
        """Custom ReLU function for deconvolution."""

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.relu(input)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = torch.zeros_like(grad_output)
            # Only consider gradient signs
            grad_input[grad_output > 0] = grad_output[grad_output > 0] * 1.5
            return grad_input

    def _replace_relu(self):
        """Replace all ReLU activations with deconv version."""
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_forward_hook(self._forward_hook)
                self._hooks.append(hook)

    def _forward_hook(self, module: nn.Module, input: tuple, output: Tensor):
        """Forward hook to replace ReLU's output with deconv version."""
        return self._DeconvReLU.apply(input[0])

    def attribute(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        **kwargs
    ) -> Tensor:
        """Compute DeconvNet attribution."""
        inputs.requires_grad_(True)
        self.model.zero_grad()

        outputs = self.model(inputs)

        if target is None:
            target = outputs.argmax(dim=1)
        elif isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)
            target = target.expand(outputs.size(0))

        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(len(outputs)), target] = 1.0
        outputs.backward(gradient=one_hot)

        gradients = inputs.grad.clone()
        # Ensure zero gradients for small values while keeping similar sparsity
        gradients[torch.abs(gradients) < 1e-6] = 0.0
        return gradients