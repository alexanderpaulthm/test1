"""Base classes for attribution methods."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

class Attribution(ABC):
    """Base class for attribution methods."""

    def __init__(self, model: nn.Module):
        """Initialize the attribution method.

        Args:
            model: The PyTorch model to analyze
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode

    @abstractmethod
    def attribute(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        **kwargs
    ) -> Tensor:
        """Compute attributions for the given inputs.

        Args:
            inputs: Input tensor to analyze
            target: Target class or output tensor
            **kwargs: Additional method-specific parameters

        Returns:
            Attribution scores for the input
        """
        pass

class GradientAttribution(Attribution):
    """Base class for gradient-based attribution methods."""

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def _get_gradients(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute gradients with respect to inputs.

        Args:
            inputs: Input tensor
            target: Target class or output tensor

        Returns:
            Tuple of (gradients, outputs)
        """
        inputs.requires_grad_(True)

        outputs = self.model(inputs)

        if target is None:
            target = outputs.argmax(dim=1)

        if isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)

        if target.dim() == 1:
            target = nn.functional.one_hot(
                target, num_classes=outputs.size(-1)
            ).float()

        gradients = torch.autograd.grad(
            (outputs * target).sum(),
            inputs,
            create_graph=True
        )[0]

        return gradients, outputs