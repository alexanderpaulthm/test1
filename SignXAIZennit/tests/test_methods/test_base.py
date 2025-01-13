"""Tests for base gradient implementations."""

import pytest
import torch
import torch.nn as nn

from signxai_torch.methods.base import BaseGradient, InputXGradient, GradientXSign


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleNet()


@pytest.fixture
def input_tensor():
    """Create a test input tensor."""
    return torch.randn(1, 3, 32, 32)


def test_base_gradient_initialization(model):
    """Test base gradient initialization."""
    base_grad = BaseGradient(model)
    assert isinstance(base_grad.model, nn.Module)


def test_base_gradient_attribution(model, input_tensor):
    """Test basic gradient attribution."""
    base_grad = BaseGradient(model)
    attribution = base_grad.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

    # Check that there are both positive and negative gradients
    assert torch.any(attribution > 0)
    assert torch.any(attribution < 0)


def test_base_gradient_with_target(model, input_tensor):
    """Test gradient attribution with specific target."""
    base_grad = BaseGradient(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = base_grad.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_base_gradient_batch_processing(model):
    """Test gradient attribution with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    base_grad = BaseGradient(model)
    attribution = base_grad.attribute(inputs)

    assert attribution.shape == inputs.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_input_x_gradient_attribution(model, input_tensor):
    """Test input times gradient attribution."""
    input_grad = InputXGradient(model)
    attribution = input_grad.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

    # Verify that attribution is input times gradient
    base_grad = BaseGradient(model)
    base_attribution = base_grad.attribute(input_tensor)
    assert torch.allclose(attribution, input_tensor * base_attribution)


def test_gradient_x_sign_attribution(model, input_tensor):
    """Test gradient times SIGN attribution."""
    grad_sign = GradientXSign(model)
    attribution = grad_sign.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

    # Verify that gradient sign is properly applied
    base_grad = BaseGradient(model)
    base_attribution = base_grad.attribute(input_tensor)
    pos_mask = base_attribution > 0
    neg_mask = base_attribution < 0
    assert torch.all(attribution[pos_mask] >= 0)
    assert torch.all(attribution[neg_mask] <= 0)


def test_gradient_x_sign_mu_parameter(model, input_tensor):
    """Test SIGN mu parameter effect."""
    mu_values = [-0.5, 0.0, 0.5]
    attributions = []

    for mu in mu_values:
        grad_sign = GradientXSign(model, mu=mu)
        attribution = grad_sign.attribute(input_tensor)
        attributions.append(attribution)

    # Check that different mu values produce different attributions
    assert not torch.allclose(attributions[0], attributions[1])
    assert not torch.allclose(attributions[1], attributions[2])


def test_numerical_stability(model):
    """Test numerical stability with edge cases."""
    # Test with very small values
    small_input = torch.randn(1, 3, 32, 32) * 1e-5

    # Base gradient
    base_grad = BaseGradient(model)
    small_attr = base_grad.attribute(small_input)
    assert not torch.isnan(small_attr).any()

    # Input x gradient
    input_grad = InputXGradient(model)
    small_attr = input_grad.attribute(small_input)
    assert not torch.isnan(small_attr).any()

    # Gradient x SIGN
    grad_sign = GradientXSign(model)
    small_attr = grad_sign.attribute(small_input)
    assert not torch.isnan(small_attr).any()


def test_error_handling(model, input_tensor):
    """Test error handling with invalid inputs."""
    base_grad = BaseGradient(model)

    # Test with invalid target class
    with pytest.raises(Exception):
        base_grad.attribute(input_tensor, target=torch.tensor([100]))  # Invalid class

    # Test with invalid target shape
    with pytest.raises(Exception):
        base_grad.attribute(input_tensor, target=torch.tensor([[1, 2]]))  # Wrong shape

    # Test with incompatible batch size
    with pytest.raises(Exception):
        base_grad.attribute(input_tensor, target=torch.tensor([1, 2]))  # Too many targets