"""Tests for SmoothGrad implementation."""

import pytest
import torch
import torch.nn as nn

from signxai_torch.methods.smoothgrad import SmoothGrad, VarGrad


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


def test_smoothgrad_initialization(model):
    """Test SmoothGrad initialization."""
    smoothgrad = SmoothGrad(model)
    assert smoothgrad.n_samples == 50  # Default value
    assert smoothgrad.noise_level == 0.2  # Default value
    assert smoothgrad.batch_size == 50  # Default value

    # Test custom parameters
    smoothgrad = SmoothGrad(model, n_samples=100, noise_level=0.1, batch_size=10)
    assert smoothgrad.n_samples == 100
    assert smoothgrad.noise_level == 0.1
    assert smoothgrad.batch_size == 10


def test_smoothgrad_attribution(model, input_tensor):
    """Test basic SmoothGrad attribution."""
    smoothgrad = SmoothGrad(model)
    attribution = smoothgrad.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

    # Check that there are both positive and negative gradients
    assert torch.any(attribution > 0)
    assert torch.any(attribution < 0)


def test_smoothgrad_with_target(model, input_tensor):
    """Test SmoothGrad with specific target."""
    smoothgrad = SmoothGrad(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = smoothgrad.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_smoothgrad_batch_processing(model):
    """Test SmoothGrad with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    smoothgrad = SmoothGrad(model)
    attribution = smoothgrad.attribute(inputs)

    assert attribution.shape == inputs.shape


def test_vargrad_attribution(model, input_tensor):
    """Test VarGrad attribution."""
    vargrad = VarGrad(model)
    attribution = vargrad.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Variance should be non-negative
    assert torch.all(attribution >= 0)

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_noise_scaling(model, input_tensor):
    """Test that noise scaling works properly."""
    # Test with different noise levels
    noise_levels = [0.1, 0.2, 0.5]
    attributions = []

    for noise_level in noise_levels:
        smoothgrad = SmoothGrad(model, noise_level=noise_level)
        attribution = smoothgrad.attribute(input_tensor)
        attributions.append(attribution)

    # Higher noise should lead to more variance in attributions
    var_low = torch.var(attributions[0])
    var_high = torch.var(attributions[-1])
    assert var_high > var_low, "Higher noise should lead to more variance"


def test_numerical_stability(model):
    """Test numerical stability with edge cases."""
    # Test with very small values
    small_input = torch.randn(1, 3, 32, 32) * 1e-5
    smoothgrad = SmoothGrad(model)
    small_attr = smoothgrad.attribute(small_input)
    assert not torch.isnan(small_attr).any()

    # Test with very large values
    large_input = torch.randn(1, 3, 32, 32) * 1e5
    large_attr = smoothgrad.attribute(large_input)
    assert not torch.isnan(large_attr).any()