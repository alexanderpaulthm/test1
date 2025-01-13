"""Tensor operation utilities."""

from typing import Optional, Union, Tuple, List

import torch
from torch import Tensor
import numpy as np


def standardize_tensor(
        tensor: Tensor,
        mean: Optional[Union[float, Tensor]] = None,
        std: Optional[Union[float, Tensor]] = None,
        eps: float = 1e-8,
        dim: Optional[Union[int, List[int]]] = None
) -> Tensor:
    """Standardize a tensor."""
    if tensor.numel() == 0:
        raise RuntimeError("Cannot standardize empty tensor")

    # Convert to double precision for stability
    tensor = tensor.to(dtype=torch.float64).detach()

    # Compute or use provided mean
    if mean is None:
        mean = tensor.mean(dim=dim, keepdim=True if dim is not None else False)
    elif not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, dtype=torch.float64, device=tensor.device)

    # Center the data
    result = tensor - mean

    # Compute or use provided std
    if std is None:
        std = result.std(dim=dim, keepdim=True if dim is not None else False)
    elif not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=torch.float64, device=tensor.device)

    # Handle constant tensors
    if (isinstance(std, torch.Tensor) and (std < eps).all()) or (not isinstance(std, torch.Tensor) and std < eps):
        return torch.zeros_like(tensor, dtype=torch.float32)

    # Standardize
    result = result / (std + eps)

    # Convert back to float32
    return result.to(dtype=torch.float32)

def normalize_tensor(
    tensor: Tensor,
    min_val: Optional[Union[float, Tensor]] = None,
    max_val: Optional[Union[float, Tensor]] = None,
    out_range: Tuple[float, float] = (0.0, 1.0)
) -> Tensor:
    """Normalize a tensor to a specified range."""
    if tensor.numel() == 0:
        raise RuntimeError("Cannot normalize empty tensor")

    tensor = tensor.float()

    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()

    # Handle constant tensors
    if torch.allclose(min_val * torch.ones(1), max_val * torch.ones(1)):
        return torch.full_like(tensor, out_range[0])

    # Normalize to [0, 1]
    normalized = (tensor - min_val) / (max_val - min_val + 1e-7)

    # Scale to output range
    normalized = normalized * (out_range[1] - out_range[0]) + out_range[0]
    return normalized.clamp(out_range[0], out_range[1])

def preprocess_image(
    image: Union[np.ndarray, Tensor],
    mean: Optional[Union[float, Tuple[float, ...], List[float]]] = None,
    std: Optional[Union[float, Tuple[float, ...], List[float]]] = None,
    device: Optional[torch.device] = None
) -> Tensor:
    """Preprocess image for model input."""
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if image.ndim not in [2, 3, 4]:
            raise ValueError(f"Invalid image dimensions: {image.ndim}")

        if image.ndim == 3 and image.shape[-1] not in [1, 3] and image.shape[0] not in [1, 3]:
            raise ValueError(f"Invalid number of channels: {image.shape[-1]}")

        image = torch.from_numpy(image)

    image = image.float()
    if image.max() > 1.0:
        image = image / 255.0

    if device is not None:
        image = image.to(device)

    # Handle different formats
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        if image.shape[-1] in [1, 3]:  # HWC format
            image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
    elif image.dim() == 4 and image.shape[-1] in [1, 3]:  # NHWC format
        image = image.permute(0, 3, 1, 2)

    if mean is not None and std is not None:
        if isinstance(mean, (int, float)):
            mean = [mean] * image.shape[1]
        if isinstance(std, (int, float)):
            std = [std] * image.shape[1]

        mean = torch.tensor(mean, device=image.device, dtype=image.dtype).view(-1, 1, 1)
        std = torch.tensor(std, device=image.device, dtype=image.dtype).view(-1, 1, 1)
        image = (image - mean) / (std + 1e-7)

    return image

def deprocess_tensor(
    tensor: Tensor,
    mean: Optional[Union[float, Tuple[float, ...], List[float]]] = None,
    std: Optional[Union[float, Tuple[float, ...], List[float]]] = None,
    denormalize: bool = True,
    to_numpy: bool = False
) -> Union[Tensor, np.ndarray]:
    """Convert model output tensor back to image format."""
    if tensor.dim() not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.dim()}D")

    tensor = tensor.detach().clone()

    if tensor.dim() == 4:
        if tensor.size(0) > 1:
            tensor = tensor[0]
        else:
            tensor = tensor.squeeze(0)

    if denormalize and mean is not None and std is not None:
        if isinstance(mean, (int, float)):
            mean = [mean] * tensor.shape[0]
        if isinstance(std, (int, float)):
            std = [std] * tensor.shape[0]

        mean = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(-1, 1, 1)
        tensor = tensor * std + mean

    tensor = tensor.clamp(0, 1)

    if to_numpy:
        tensor = tensor.cpu().numpy()
        if tensor.shape[0] in [1, 3]:
            tensor = np.transpose(tensor, (1, 2, 0))
            if tensor.shape[-1] == 1:
                tensor = np.squeeze(tensor, axis=-1)

    return tensor