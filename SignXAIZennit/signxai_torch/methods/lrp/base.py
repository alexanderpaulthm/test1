"""Base LRP implementation."""

from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
from torch import Tensor

from ...core.attribution import GradientAttribution


class LRP(GradientAttribution):
    """Base Layer-wise Relevance Propagation implementation."""

    def __init__(
            self,
            model: nn.Module,
            layer_rules: Optional[Dict[str, str]] = None
    ):
        super().__init__(model)
        self.layer_rules = layer_rules or {}
        self._hooks = []
        self._relevances = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks for relevance computation."""
        # Clear any existing hooks
        self._remove_hooks()
        self._hooks = []

        # Count layers that need hooks
        hook_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                hook_layers.append(module)

        # Register hooks for each layer
        for module in hook_layers:
            # Always register both forward and backward hooks
            self._hooks.append(
                module.register_forward_hook(self._forward_hook)
            )
            self._hooks.append(
                module.register_full_backward_hook(self._backward_hook)
            )

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _forward_hook(self, module: nn.Module, input: tuple, output: Tensor):
        """Store layer inputs and outputs for relevance computation."""
        self._relevances[module] = {
            'input': input[0].detach(),
            'output': output.detach()
        }

    def _backward_hook(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        """Apply LRP rules during backward pass."""
        if module not in self.layer_rules:
            return grad_input

        rule = self.layer_rules[module]
        relevance = self._compute_relevance(
            module,
            self._relevances[module]['input'],
            self._relevances[module]['output'],
            grad_output[0],
            rule
        )
        return (relevance,) + grad_input[1:]

    def _compute_relevance(
            self,
            module: nn.Module,
            input: Tensor,
            output: Tensor,
            relevance: Tensor,
            rule: str
    ) -> Tensor:
        """Compute layer relevance using specified rule."""
        raise NotImplementedError(
            "Base LRP class does not implement relevance computation. "
            "Use a specific LRP variant."
        )

    def _validate_target(self, target: Union[int, Tensor], outputs: Tensor):
        """Validate target format and value."""
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
        """Compute LRP attribution."""
        inputs.requires_grad_(True)
        self._relevances.clear()

        outputs = self.model(inputs)
        if target is None:
            target = outputs.argmax(dim=1)
        else:
            target = self._validate_target(target, outputs)

        batch_size = outputs.size(0)
        one_hot = torch.zeros_like(outputs)
        one_hot[torch.arange(batch_size), target] = 1.0
        outputs.backward(gradient=one_hot)

        return inputs.grad

    def __del__(self):
        """Clean up hooks when object is deleted."""
        self._remove_hooks()