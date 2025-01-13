"""Alpha-Beta LRP implementation."""
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import LRP


class AlphaBetaLRP(LRP):
    """Alpha-Beta LRP implementation."""

    def __init__(
            self,
            model: nn.Module,
            alpha: float = 1.0,
            beta: float = 0.0,
            epsilon: float = 1e-7,
            layer_rules: Optional[Dict[str, str]] = None
    ):
        super().__init__(model, layer_rules)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        assert alpha - beta == 1.0, "Alpha - Beta must equal 1"

    def _reshape_relevance(self, relevance: Tensor, shape: tuple) -> Tensor:
        """Reshape relevance tensor to match required shape."""
        if relevance.shape != shape:
            if len(shape) == 4 and len(relevance.shape) == 2:
                batch, channels = relevance.shape
                _, _, height, width = shape
                return relevance.view(batch, channels, 1, 1).expand(-1, -1, height, width)
            elif len(shape) == 2 and len(relevance.shape) == 4:
                return relevance.mean(dim=(2, 3))
        return relevance

    def _normalize_relevance(self, relevance: Tensor, reference: Tensor) -> Tensor:
        """Normalize relevance to match reference sum."""
        ref_sum = reference.sum()
        rel_sum = relevance.sum()

        if rel_sum.abs() > self.epsilon:  # Avoid division by zero
            return relevance * (ref_sum / rel_sum)
        return relevance

    def _compute_relevance(
            self,
            module: nn.Module,
            input: Tensor,
            output: Tensor,
            relevance: Tensor,
            rule: str
    ) -> Tensor:
        """Compute layer relevance using alpha-beta rule."""
        if not isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            return relevance

        # Ensure input requires grad
        input.requires_grad_(True)

        # Reshape relevance to match output shape
        relevance = self._reshape_relevance(relevance, output.shape)

        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weights = module.weight
            stabilizer = self.epsilon * (torch.abs(output).max() + self.epsilon)

            # Split weights into positive and negative parts
            weights_pos = torch.clamp(weights, min=0)
            weights_neg = torch.clamp(weights, max=0)

            # Split activations
            z = output.clone()
            z_pos = F.relu(z)
            z_neg = z - z_pos

            # Compute denominators with stabilizer
            denom_pos = z_pos + stabilizer
            denom_neg = z_neg - stabilizer

            if isinstance(module, nn.Conv2d):
                # Calculate output padding
                out_pad = self._calculate_output_padding(input.shape, output.shape, module)

                # Compute positive contribution
                pos_relevance = F.conv_transpose2d(
                    relevance * (z_pos / denom_pos) * self.alpha,
                    weights_pos,
                    padding=module.padding,
                    stride=module.stride,
                    output_padding=out_pad
                )

                # Compute negative contribution
                neg_relevance = F.conv_transpose2d(
                    relevance * (z_neg / denom_neg) * self.beta,
                    weights_neg,
                    padding=module.padding,
                    stride=module.stride,
                    output_padding=out_pad
                )
            else:  # Linear layer
                pos_relevance = F.linear(
                    relevance * (z_pos / denom_pos) * self.alpha,
                    weights_pos.t()
                )
                neg_relevance = F.linear(
                    relevance * (z_neg / denom_neg) * self.beta,
                    weights_neg.t()
                )

            # Combine and normalize relevance
            total_relevance = pos_relevance - neg_relevance
            return self._normalize_relevance(total_relevance, relevance)

        elif isinstance(module, nn.BatchNorm2d):
            running_std = torch.sqrt(module.running_var + module.eps)
            normalized = (input - module.running_mean.view(1, -1, 1, 1)) / running_std.view(1, -1, 1, 1)
            return relevance * normalized

        return relevance

    def _calculate_output_padding(self, input_shape, output_shape, module):
        """Calculate output padding for transposed convolution."""
        if not isinstance(module, nn.Conv2d):
            return 0

        stride = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        padding = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)

        out_pad = []
        for i in range(2):  # Height and width
            size_i = input_shape[-(2-i)]
            out_i = output_shape[-(2-i)]
            stride_i = stride[i]
            pad_i = padding[i]
            kernel_i = kernel_size[i]

            required_pad = size_i - ((out_i - 1) * stride_i - 2 * pad_i + kernel_i)
            out_pad.append(max(0, required_pad))

        return tuple(out_pad)

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute Alpha-Beta LRP attribution."""
        inputs = inputs.clone().detach().requires_grad_(True)
        self._relevances.clear()

        # Forward pass
        outputs = self.model(inputs)
        batch_size = outputs.size(0)

        # Handle target selection
        if target is None:
            target = outputs.argmax(dim=1)
        else:
            target = self._validate_target(target, outputs)

        # Initialize relevance with output scores
        relevance = torch.zeros_like(outputs)
        target_scores = outputs.gather(1, target.view(-1, 1))
        relevance.scatter_(1, target.view(-1, 1), target_scores)

        # Backward pass with relevance propagation
        for module in reversed(list(self.model.modules())):
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                if module in self._relevances:
                    relevance = self._compute_relevance(
                        module,
                        self._relevances[module]['input'],
                        self._relevances[module]['output'],
                        relevance,
                        self.layer_rules.get(module.__class__.__name__, 'alpha_beta')
                    )

        # Final normalization to match output score
        final_attr = inputs * relevance
        return self._normalize_relevance(final_attr, target_scores)