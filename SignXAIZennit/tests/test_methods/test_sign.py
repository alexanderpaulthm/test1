"""Tests for the SIGN method implementation."""

import pytest
import torch
import torch.nn as nn

from signxai_torch.methods.sign import SIGN

class SimpleConvNet(nn.Module):
    """Simple CNN for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleConvNet()

@pytest.fixture
def input_tensor():
    """Create a test input tensor."""
    return torch.randn(1, 3, 32, 32)

def test_sign_attribution(model, input_tensor):
    """Test basic SIGN attribution."""
    sign = SIGN(model)
    attribution = sign.attribute(input_tensor)

    # Check shape
    assert attribution.shape == input_tensor.shape

    # Check dtype
    assert attribution.dtype == input_tensor.dtype

    # Check device
    assert attribution.device == input_tensor.device

    # Check that attribution is not all zeros
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_target_attribution(model, input_tensor):
    """Test SIGN attribution with specific target."""
    sign = SIGN(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = sign.attribute(input_tensor, target=target)

    # Basic checks
    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_batch_attribution(model):
    """Test SIGN attribution with batched inputs."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)

    sign = SIGN(model)
    attribution = sign.attribute(inputs, batch_size=2)

    assert attribution.shape == inputs.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_sign_integrated_gradients(model, input_tensor):
    """Test integrated gradients variant of SIGN."""
    sign = SIGN(model)
    attribution = sign.get_integrated_gradients(input_tensor, steps=10)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

def test_invalid_input_handling(model):
    """Test handling of invalid inputs."""
    sign = SIGN(model)

    # Test empty input
    with pytest.raises(ValueError):
        sign.attribute(torch.tensor([]))