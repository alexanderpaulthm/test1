"""Method wrappers providing different parameter combinations."""

import warnings
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, List, Tuple
from torch import Tensor
import torch.nn as nn

from .gradcam import GradCAM
from .guided import GuidedBackprop, DeconvNet
from .base import BaseGradient, InputXGradient, GradientXSign
from .integrated import IntegratedGradients
from .smoothgrad import SmoothGrad, VarGrad
from .sign import SIGN
from .lrp.base import LRP
from .lrp.epsilon import EpsilonLRP
from .lrp.alpha_beta import AlphaBetaLRP


def _normalize_dimensions(tensor: Tensor) -> Tensor:
    """Normalize tensor dimensions ensuring 4D BCHW format."""
    if tensor.dim() == 2:  # HW
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 3:  # CHW
        return tensor.unsqueeze(0)
    elif tensor.dim() == 4:  # BCHW
        return tensor
    raise ValueError(f"Expected 2D, 3D or 4D tensor, got {tensor.dim()}D")


def _ensure_tensor_size(result: Tensor, reference: Tensor) -> Tensor:
    """Ensure tensor has same size as reference tensor."""
    if not isinstance(result, torch.Tensor):
        raise TypeError("Result must be a tensor")

    if result.shape == reference.shape:
        return result

    # Convert to 4D if needed
    if result.dim() < 4:
        result = _normalize_dimensions(result)
    if reference.dim() < 4:
        reference = _normalize_dimensions(reference)

    # Handle spatial dimensions
    if result.shape[-2:] != reference.shape[-2:]:
        result = F.interpolate(
            result,
            size=reference.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

    # Handle channel dimension
    if result.shape[1] != reference.shape[1]:
        result = result.expand(-1, reference.shape[1], -1, -1)

    # Handle batch dimension
    if result.shape[0] != reference.shape[0]:
        result = result.expand(reference.shape[0], -1, -1, -1)

    return result


def _get_layer_rules(model: nn.Module, rule_type: str = 'epsilon') -> Dict[str, str]:
    """Get layer rules mapping for model."""
    rules = {}
    conv_seen = False

    # First pass: Find all layers that need rules
    modules = list(model.named_modules())

    # Second pass: Assign rules
    for name, module in modules:
        if isinstance(module, nn.Conv2d):
            if not conv_seen:
                rules[name] = 'zb'
                conv_seen = True
            else:
                rules[name] = rule_type
        elif isinstance(module, nn.Linear):
            rules[name] = 'epsilon'

    # Final pass: Ensure no layer was missed
    for name, module in modules:
        if isinstance(module, (nn.Conv2d, nn.Linear)) and name not in rules:
            rules[name] = rule_type

    return rules


def _get_epsilon_value(method: str, inputs: Tensor) -> float:
    """Extract epsilon value from method name."""
    if "_std_x" in method:
        scale = float(method.split("_std_x")[0].split("_")[-1])
        return scale * inputs.std()
    return float(method.split("epsilon_")[-1].split("_")[0])


def _get_mu_value(method: str) -> float:
    """Extract mu value from method name."""
    if "_mu_" not in method:
        return 0.0
    mu_str = method.split("_mu_")[-1]
    return float(mu_str.replace("neg_", "-"))


def validate_model_type(model: nn.Module, expected_type: str) -> bool:
    """Validate if model is of expected type."""
    return expected_type.lower() in str(type(model).__name__).lower()


def get_default_vgg_layer(model: nn.Module) -> str:
    """Get default last conv layer for VGG models."""
    if not validate_model_type(model, "VGG"):
        raise ValueError("Model is not a VGG model")

    if not hasattr(model, 'features'):
        raise ValueError("Model does not have features attribute")

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)

    if not conv_layers:
        raise ValueError("No convolutional layers found in model")

    return conv_layers[-1]


def _handle_lrp_method(
        method: str,
        model: nn.Module,
        inputs: Tensor,
        target: Optional[Tensor] = None,
        **kwargs
) -> Optional[Tensor]:
    """Handle LRP method computation with proper error handling."""
    try:
        # Get base rule type
        if method.startswith('w2lrp_'):
            rule_type = 'w2'
        elif method.startswith('flatlrp_'):
            rule_type = 'flat'
        elif method.startswith('zblrp_'):
            rule_type = 'zb'
        else:
            rule_type = 'epsilon'

        # Get layer rules
        rules = _get_layer_rules(model, rule_type)

        # Handle VGG specific methods
        if "VGG16ILSVRC" in method and not validate_model_type(model, "VGG"):
            raise ValueError(f"Method {method} requires VGG16 model")

        # Extract parameters
        use_sign = 'sign' in method
        mu = _get_mu_value(method) if use_sign else 0.0

        # Handle different LRP variants
        if method == "lrp_z" or method == "lrpsign_z":
            return EpsilonLRP(model, epsilon=0.0, use_sign=use_sign, mu=mu, layer_rules=rules).attribute(inputs, target)

        if method.startswith(("lrp_epsilon_", "w2lrp_epsilon_", "flatlrp_epsilon_", "zblrp_epsilon_", "lrpsign_epsilon_")):
            epsilon = _get_epsilon_value(method, inputs)
            return EpsilonLRP(model, epsilon=epsilon, use_sign=use_sign, mu=mu, layer_rules=rules).attribute(inputs, target)

        if "alpha_1_beta_0" in method:
            return AlphaBetaLRP(model, alpha=1.0, beta=0.0, layer_rules=rules).attribute(inputs, target)

        if "sequential_composite" in method:
            variant = method[-1]  # 'a' or 'b'
            custom_rules = {name: ('zb' if i == 0 else 'alpha_beta' if variant == 'b' else 'epsilon')
                          for i, (name, _) in enumerate(rules.items())}
            return EpsilonLRP(model, use_sign=use_sign, mu=mu, layer_rules=custom_rules).attribute(inputs, target)

        return None

    except Exception as e:
        raise ValueError(f"Error in LRP computation: {str(e)}")


def calculate_relevancemap(
        method: str,
        inputs: Tensor,
        model: nn.Module,
        target: Optional[Union[int, Tensor]] = None,
        **kwargs
) -> Tensor:
    """Calculate relevance map using specified method."""
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")

    # Ensure proper input dimensions
    inputs = _normalize_dimensions(inputs)

    # Process target
    if target is not None:
        if isinstance(target, int):
            target = torch.tensor([target], device=inputs.device)
        elif isinstance(target, torch.Tensor):
            target = target.to(inputs.device)
            if target.dim() == 0:
                target = target.unsqueeze(0)

    try:
        # Basic gradient methods
        if method == "gradient":
            return BaseGradient(model).attribute(inputs, target)

        if method in ["input_t_gradient", "gradient_x_input"]:
            return InputXGradient(model).attribute(inputs, target)

        # Gradient x SIGN methods
        if method.startswith("gradient_x_sign"):
            mu = _get_mu_value(method)
            return GradientXSign(model, mu=mu).attribute(inputs, target)

        # Guided backprop methods
        if method.startswith("guided_backprop"):
            if "_x_sign" in method:
                mu = _get_mu_value(method)
                base_result = GuidedBackprop(model).attribute(inputs, target)
                return SIGN(model, mu=mu).attribute(base_result)
            return GuidedBackprop(model).attribute(inputs, target)

        # DeconvNet methods
        if method.startswith("deconvnet"):
            if "_x_sign" in method:
                mu = _get_mu_value(method)
                base_result = DeconvNet(model).attribute(inputs, target)
                return SIGN(model, mu=mu).attribute(base_result)
            return DeconvNet(model).attribute(inputs, target)

        # SmoothGrad methods
        if method.startswith(("smoothgrad", "vargrad")):
            n_samples = kwargs.get('n_samples', 50)
            noise_level = kwargs.get('noise_level', 0.2)
            base_method = SmoothGrad if "smoothgrad" in method else VarGrad
            if "_x_sign" in method:
                mu = _get_mu_value(method)
                base_result = base_method(model, n_samples=n_samples, noise_level=noise_level).attribute(inputs, target)
                return SIGN(model, mu=mu).attribute(base_result)
            return base_method(model, n_samples=n_samples, noise_level=noise_level).attribute(inputs, target)

        # GradCAM methods
        if method.startswith("grad_cam"):
            if method == "grad_cam_VGG16ILSVRC":
                if not validate_model_type(model, "VGG"):
                    raise ValueError("Model must be VGG type for grad_cam_VGG16ILSVRC")
                last_conv = get_default_vgg_layer(model)
            else:
                last_conv = kwargs.get('last_conv')
                if not last_conv:
                    raise ValueError("last_conv layer must be specified for GradCAM")

            if isinstance(last_conv, str):
                modules = dict(model.named_modules())
                if last_conv not in modules:
                    raise ValueError(f"Layer {last_conv} not found in model")
                last_conv = modules[last_conv]

            result = GradCAM(model, last_conv).attribute(inputs, target)
            if result.dim() == 3:
                result = result.unsqueeze(1)
            if result.size(1) != inputs.size(1):
                result = result.expand(result.size(0), inputs.size(1), result.size(2), result.size(3))
            return result

        # Integrated gradients
        if method == "integrated_gradients":
            steps = kwargs.get('steps', 50)
            baseline = kwargs.get('baseline', None)
            return IntegratedGradients(model, steps=steps).attribute(inputs, target, baseline=baseline)

        # Handle LRP methods
        result = _handle_lrp_method(method, model, inputs, target, **kwargs)
        if result is not None:
            return result

        raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise RuntimeError(f"Error computing relevance map: {str(e)}") from e