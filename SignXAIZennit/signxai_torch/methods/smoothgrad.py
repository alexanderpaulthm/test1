from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..core.attribution import GradientAttribution


class SmoothGrad(GradientAttribution):
    """SmoothGrad implementation."""

    def __init__(
            self,
            model: nn.Module,
            n_samples: int = 50,
            noise_level: float = 0.2,
            batch_size: Optional[int] = None
    ):
        super().__init__(model)
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.batch_size = batch_size if batch_size is not None else n_samples

    def _create_target_vector(self, outputs: Tensor, target: Optional[Union[int, Tensor]], batch_size: int) -> Tensor:
        """Create target vector for gradient computation."""
        if target is None:
            target = outputs.max(dim=1)[1]
        elif isinstance(target, int):
            target = torch.full((batch_size,), target, device=outputs.device)
        elif isinstance(target, torch.Tensor) and target.numel() == 1:
            target = target.repeat(batch_size)
        elif len(target) != batch_size:
            target = target[:batch_size]  # Trim to batch size
        return target

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute SmoothGrad attribution."""
        inputs.requires_grad_(True)

        # Calculate input range for noise scaling
        input_max = inputs.max()
        input_min = inputs.min()
        input_range = input_max - input_min
        noise_scale = self.noise_level * input_range

        # Initialize cumulative gradients
        cumulative_grads = torch.zeros_like(inputs)
        input_is_batch = inputs.dim() == 4 and inputs.shape[0] > 1

        # Process in batches
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            curr_batch_size = end_idx - start_idx

            # Create noisy samples
            if input_is_batch:
                base_inputs = inputs
            else:
                base_inputs = inputs.repeat(curr_batch_size, 1, 1, 1)

            noise = torch.randn_like(base_inputs) * noise_scale
            noisy_inputs = base_inputs + noise
            noisy_inputs.retain_grad()  # Ensure gradients are retained

            # Forward pass
            outputs = self.model(noisy_inputs)
            batch_target = self._create_target_vector(outputs, target, outputs.shape[0])

            # Gradient computation
            self.model.zero_grad()
            loss = torch.sum(outputs * torch.nn.functional.one_hot(batch_target, outputs.shape[1]).float())
            loss.backward()

            # Accumulate gradients
            if noisy_inputs.grad is not None:
                grad = noisy_inputs.grad
                if not input_is_batch:
                    grad = grad.mean(dim=0, keepdim=True)
                cumulative_grads += grad.detach() * noise_scale

        return cumulative_grads / self.n_samples


class VarGrad(SmoothGrad):
    """VarGrad implementation."""

    def attribute(
            self,
            inputs: Tensor,
            target: Optional[Union[int, Tensor]] = None,
            **kwargs
    ) -> Tensor:
        """Compute VarGrad attribution."""
        inputs.requires_grad_(True)

        # Calculate noise scale
        input_range = inputs.max() - inputs.min()
        noise_scale = self.noise_level * input_range

        # Store all gradients
        all_grads = []
        input_is_batch = inputs.dim() == 4 and inputs.shape[0] > 1

        # Process in batches
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            curr_batch_size = end_idx - start_idx

            # Create noisy samples
            if input_is_batch:
                base_inputs = inputs
            else:
                base_inputs = inputs.repeat(curr_batch_size, 1, 1, 1)

            noise = torch.randn_like(base_inputs) * noise_scale
            noisy_inputs = base_inputs + noise
            noisy_inputs.retain_grad()  # Ensure gradients are retained

            # Forward pass
            outputs = self.model(noisy_inputs)
            batch_target = self._create_target_vector(outputs, target, outputs.shape[0])

            # Gradient computation
            self.model.zero_grad()
            loss = torch.sum(outputs * torch.nn.functional.one_hot(batch_target, outputs.shape[1]).float())
            loss.backward()

            # Store gradients
            if noisy_inputs.grad is not None:
                grad = noisy_inputs.grad * noise_scale  # Scale gradients by noise level
                if not input_is_batch:
                    grad = grad.mean(dim=0, keepdim=True)
                all_grads.append(grad.detach())

        if len(all_grads) > 0:
            all_grads = torch.cat(all_grads, dim=0)
            # Compute variance and ensure shape matches input shape by adding batch dimension if needed
            variance = torch.var(all_grads, dim=0, unbiased=False)
            if not input_is_batch and variance.dim() == 3:
                variance = variance.unsqueeze(0)
            # Ensure non-zero variance by adding small epsilon
            epsilon = 1e-7
            variance = variance + epsilon
            return variance
        else:
            return torch.zeros_like(inputs)