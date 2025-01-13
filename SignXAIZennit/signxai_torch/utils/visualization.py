"""Visualization utilities for attribution methods."""

from typing import Optional, Union, Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor


def project(
        X: Union[Tensor, np.ndarray],
        output_range: Tuple[float, float] = (0, 1),
        absmax: Optional[np.ndarray] = None,
        input_is_positive_only: bool = False,
        **kwargs  # Accept additional kwargs for compatibility
) -> np.ndarray:
    """Projects a tensor into a specified value range."""
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if absmax is None:
        absmax = np.max(np.abs(X), axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    if mask.sum() > 0:
        X = np.divide(X, absmax[mask], out=np.copy(X), where=mask)

    if not input_is_positive_only:
        X = (X + 1) / 2
    X = np.clip(X, 0, 1)

    return output_range[0] + (X * (output_range[1] - output_range[0]))


def heatmap(
        X: Union[Tensor, np.ndarray],
        cmap_type: str = "seismic",
        reduce_op: str = "sum",
        reduce_axis: int = -1,
        alpha_cmap: bool = False,
        **kwargs
) -> np.ndarray:
    """Creates a heatmap visualization."""
    cmap = plt.cm.get_cmap(cmap_type)

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if reduce_op == "sum":
        X = X.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = X.max(axis=reduce_axis)
        neg_max = (-X).max(axis=reduce_axis)
        X = np.where(pos_max >= -neg_max, pos_max, neg_max)
    else:
        raise ValueError(f"Unknown reduce operation: {reduce_op}")

    # Filter out parameters not used by project
    project_kwargs = {
        'output_range': kwargs.get('output_range', (0, 1)),
        'absmax': kwargs.get('absmax', None),
        'input_is_positive_only': kwargs.get('input_is_positive_only', False)
    }

    X = project(X, **project_kwargs)

    if alpha_cmap:
        return cmap(X)
    return cmap(X)[:, :3]


def normalize_attribution(
        attribution: Union[Tensor, np.ndarray],
        percentile: Optional[float] = None,
        symmetric: bool = False
) -> np.ndarray:
    """Normalize attribution scores."""
    if isinstance(attribution, Tensor):
        attribution = attribution.detach().cpu().numpy()

    if attribution.size == 0:
        raise ValueError("Empty attribution array")

    # Handle constant arrays
    if np.allclose(attribution, attribution.flat[0]):
        const_val = attribution.flat[0]
        if np.isclose(const_val, 1.0):
            return np.ones_like(attribution)
        if np.isclose(const_val, 0.0):
            return np.zeros_like(attribution)
        if symmetric:
            return np.zeros_like(attribution)
        return np.ones_like(attribution) if const_val > 0 else np.zeros_like(attribution)

    # Apply percentile-based clipping
    if percentile is not None:
        if not 0 <= percentile <= 100:
            raise ValueError("Percentile must be between 0 and 100")
        lower = np.percentile(attribution, (100 - percentile) / 2)
        upper = np.percentile(attribution, 100 - (100 - percentile) / 2)
        attribution = np.clip(attribution, lower, upper)

    # Normalize
    if symmetric:
        abs_max = np.abs(attribution).max()
        if abs_max > 0:
            return attribution / abs_max
        return attribution
    else:
        attr_min, attr_max = attribution.min(), attribution.max()
        if attr_max > attr_min:
            return (attribution - attr_min) / (attr_max - attr_min)
        return attribution.copy()


def _prepare_image(image: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Prepare image for visualization."""
    if isinstance(image, Tensor):
        image = image.detach().cpu().numpy()

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, -1)
    elif image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        if image.shape[0] == 1:
            image = image[0]
            if image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))
        else:
            raise ValueError("Expected single image, got batch size > 1")

    return image


def visualize_attribution(
        image: Union[Tensor, np.ndarray],
        attribution: Union[Tensor, np.ndarray],
        cmap: str = 'seismic',
        alpha: float = 0.5,
        percentile: Optional[float] = None,
        show_colorbar: bool = True
) -> plt.Figure:
    """Visualize attribution overlay on image."""
    image = _prepare_image(image)

    if isinstance(attribution, Tensor):
        attribution = attribution.detach().cpu().numpy()

    if attribution.ndim == 3 and attribution.shape[0] in [1, 3]:
        attribution = attribution[0]
    elif attribution.ndim == 4:
        attribution = attribution[0, 0]

    if image.shape[:2] != attribution.shape[:2]:
        raise ValueError("Image and attribution shapes must match")

    norm_image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Use same kwargs for consistency
    project_kwargs = {
        'output_range': (0, 1),
        'input_is_positive_only': False
    }

    hm = heatmap(attribution, cmap_type=cmap, **project_kwargs)

    fig, ax = plt.subplots()
    ax.imshow(norm_image)
    im = ax.imshow(hm, cmap=cmap, alpha=alpha)

    if show_colorbar:
        plt.colorbar(im, ax=ax)

    ax.axis('off')
    plt.tight_layout()
    return fig


def visualize_multiple_attributions(
        image: Union[Tensor, np.ndarray],
        attributions: Dict[str, Union[Tensor, np.ndarray]],
        cmap: str = 'seismic',
        alpha: float = 0.5,
        percentile: Optional[float] = None,
        figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """Visualize multiple attribution methods."""
    if not attributions:
        raise ValueError("Empty attributions dictionary")

    # Prepare image first
    image = _prepare_image(image)
    image_shape = image.shape[:2]

    # Validate all attributions before processing
    for method, attr in attributions.items():
        if isinstance(attr, Tensor):
            attr = attr.detach().cpu().numpy()

        if attr.ndim == 3 and attr.shape[0] in [1, 3]:
            attr = attr[0]
        elif attr.ndim == 4:
            attr = attr[0, 0]

        if attr.shape[:2] != image_shape:
            raise ValueError(
                f"Attribution shape {attr.shape[:2]} for method '{method}' "
                f"doesn't match image shape {image_shape}"
            )

    # Rest of the visualization code...
    n_methods = len(attributions)
    cols = min(3, n_methods)
    rows = (n_methods + cols - 1) // cols
    figsize = figsize or (5 * cols, 5 * rows)

    norm_image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (method, attr) in enumerate(attributions.items()):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        # Use same kwargs for consistency
        project_kwargs = {
            'output_range': (0, 1),
            'input_is_positive_only': False
        }

        hm = heatmap(attr, cmap_type=cmap, **project_kwargs)

        ax.imshow(norm_image)
        im = ax.imshow(hm, cmap=cmap, alpha=alpha)
        plt.colorbar(im, ax=ax)

        ax.set_title(method)
        ax.axis('off')

    for idx in range(len(attributions), rows * cols):
        row, col = idx // cols, idx % cols
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    return fig