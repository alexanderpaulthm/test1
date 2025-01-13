"""Tests for LRP implementations."""

import pytest
import torch
import torch.nn as nn

from signxai_torch.methods.lrp.base import LRP
from signxai_torch.methods.lrp.epsilon import EpsilonLRP
from signxai_torch.methods.lrp.alpha_beta import AlphaBetaLRP


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
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


# Base LRP Tests
def test_lrp_initialization(model):
    """Test LRP initialization."""
    lrp = LRP(model)
    assert isinstance(lrp.model, nn.Module)
    assert len(lrp._hooks) > 0  # Should have registered hooks


def test_lrp_hooks_registration(model):
    """Test that hooks are properly registered."""
    lrp = LRP(model)

    # Count number of layers that should have hooks
    expected_hook_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            expected_hook_count += 2  # Forward and backward hooks

    assert len(lrp._hooks) == expected_hook_count


def test_lrp_attribution(model, input_tensor):
    """Test basic LRP attribution."""
    lrp = LRP(model)
    attribution = lrp.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_lrp_with_target(model, input_tensor):
    """Test LRP attribution with specific target."""
    lrp = LRP(model)
    target = torch.tensor([5])  # Choose class 5
    attribution = lrp.attribute(input_tensor, target=target)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_lrp_batch_processing(model):
    """Test LRP with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    lrp = LRP(model)
    attribution = lrp.attribute(inputs)

    assert attribution.shape == inputs.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_lrp_cleanup(model):
    """Test that hooks are properly cleaned up."""
    lrp = LRP(model)
    # Count number of layers that should have hooks
    expected_hook_count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            expected_hook_count += 2  # Forward and backward hooks

    del lrp

    # Create new LRP instance - should be able to register hooks again
    new_lrp = LRP(model)
    assert len(new_lrp._hooks) == expected_hook_count


def test_lrp_layer_rules(model, input_tensor):
    """Test LRP with custom layer rules."""
    # Define custom rules for specific layers
    rules = {
        'conv1': 'epsilon',
        'conv2': 'alpha_beta'
    }

    lrp = LRP(model, layer_rules=rules)
    attribution = lrp.attribute(input_tensor)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_lrp_relevance_clearing(model, input_tensor):
    """Test that relevances are cleared between runs."""
    lrp = LRP(model)

    # First attribution
    attribution1 = lrp.attribute(input_tensor)
    stored_relevances1 = len(lrp._relevances)

    # Second attribution
    attribution2 = lrp.attribute(input_tensor)
    stored_relevances2 = len(lrp._relevances)

    assert stored_relevances1 == stored_relevances2  # Should clear between runs


def test_lrp_numerical_stability(model):
    """Test numerical stability with edge cases."""
    # Test with very small values
    small_input = torch.randn(1, 3, 32, 32) * 1e-5
    lrp = LRP(model)
    small_attr = lrp.attribute(small_input)
    assert not torch.isnan(small_attr).any()

    # Test with very large values
    large_input = torch.randn(1, 3, 32, 32) * 1e5
    large_attr = lrp.attribute(large_input)
    assert not torch.isnan(large_attr).any()


def test_lrp_error_handling(model, input_tensor):
    """Test error handling with invalid inputs."""
    lrp = LRP(model)

    # Test with invalid target class
    with pytest.raises(ValueError):
        lrp.attribute(input_tensor, target=torch.tensor([100]))  # Invalid class

    # Test with invalid target shape
    with pytest.raises(ValueError):
        lrp.attribute(input_tensor, target=torch.tensor([[1, 2]]))  # Wrong shape


# Epsilon-LRP Tests
def test_epsilon_lrp_initialization(model):
    """Test Epsilon-LRP initialization."""
    epsilon = 1e-7
    lrp = EpsilonLRP(model, epsilon=epsilon)
    assert isinstance(lrp.model, nn.Module)
    assert len(lrp._hooks) > 0
    assert lrp.epsilon == epsilon


def test_epsilon_lrp_attribution(model, input_tensor):
    """Test Epsilon-LRP attribution."""
    lrp = EpsilonLRP(model)
    attribution = lrp.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))

    # Test with different epsilon values
    for epsilon in [1e-7, 1e-5, 1e-3]:
        lrp = EpsilonLRP(model, epsilon=epsilon)
        attribution = lrp.attribute(input_tensor)
        assert not torch.isnan(attribution).any()


def test_epsilon_lrp_batch_processing(model):
    """Test Epsilon-LRP with batch input."""
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 32, 32)
    lrp = EpsilonLRP(model)
    attribution = lrp.attribute(inputs)

    assert attribution.shape == inputs.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_epsilon_lrp_numerical_stability(model, input_tensor):
    """Test Epsilon-LRP numerical stability."""
    # Test with very small values
    small_input = torch.randn(1, 3, 32, 32) * 1e-5
    lrp = EpsilonLRP(model, epsilon=1e-7)
    small_attr = lrp.attribute(small_input)
    assert not torch.isnan(small_attr).any()

    # Test with very large values
    large_input = torch.randn(1, 3, 32, 32) * 1e5
    large_attr = lrp.attribute(large_input)
    assert not torch.isnan(large_attr).any()

    # Test with different epsilon values
    for epsilon in [1e-7, 1e-5, 1e-3]:
        lrp = EpsilonLRP(model, epsilon=epsilon)
        attr = lrp.attribute(input_tensor)
        assert not torch.isnan(attr).any()
        assert not torch.isinf(attr).any()


def test_epsilon_lrp_layer_rules(model, input_tensor):
    """Test Epsilon-LRP with custom layer rules."""
    rules = {
        'conv1': 'epsilon',
        'conv2': 'epsilon'
    }

    lrp = EpsilonLRP(model, epsilon=1e-7, layer_rules=rules)
    attribution = lrp.attribute(input_tensor)

    assert attribution.shape == input_tensor.shape
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_epsilon_comparison(model, input_tensor):
    """Test that different epsilon values produce different results."""
    lrp1 = EpsilonLRP(model, epsilon=1e-7)
    lrp2 = EpsilonLRP(model, epsilon=1e-3)

    attr1 = lrp1.attribute(input_tensor)
    attr2 = lrp2.attribute(input_tensor)

    # Results should be different for different epsilon values
    assert not torch.allclose(attr1, attr2)


def test_alphabeta_lrp_initialization(model):
    """Test Alpha-Beta LRP initialization."""
    alpha, beta = 2.0, 1.0
    lrp = AlphaBetaLRP(model, alpha=alpha, beta=beta)
    assert isinstance(lrp.model, nn.Module)
    assert len(lrp._hooks) > 0
    assert lrp.alpha == alpha
    assert lrp.beta == beta


def test_alphabeta_lrp_attribution(model, input_tensor):
    """Test Alpha-Beta LRP attribution."""
    lrp = AlphaBetaLRP(model)
    attribution = lrp.attribute(input_tensor)

    # Check shape matches input
    assert attribution.shape == input_tensor.shape

    # Check that attributions are not all zero
    assert not torch.allclose(attribution, torch.zeros_like(attribution))


def test_alphabeta_lrp_values(model, input_tensor):
    """Test different alpha-beta values produce different results."""
    lrp1 = AlphaBetaLRP(model, alpha=2.0, beta=1.0)
    lrp2 = AlphaBetaLRP(model, alpha=1.5, beta=0.5)

    attr1 = lrp1.attribute(input_tensor)
    attr2 = lrp2.attribute(input_tensor)

    # Results should be different for different alpha-beta values
    assert not torch.allclose(attr1, attr2)


def test_alphabeta_lrp_conservation(model, input_tensor):
    """Test conservation property of Alpha-Beta LRP."""
    lrp = AlphaBetaLRP(model)
    attribution = lrp.attribute(input_tensor)

    # Sum of attributions should approximately equal the output
    outputs = model(input_tensor)
    target_class = outputs.argmax(dim=1)

    input_sum = attribution.sum()
    output_sum = outputs[0, target_class]

    assert torch.isclose(input_sum, output_sum, rtol=1e-2)


def test_alphabeta_lrp_invalid_params(model):
    """Test that invalid alpha-beta combinations raise error."""
    with pytest.raises(AssertionError):
        AlphaBetaLRP(model, alpha=1.0, beta=1.0)  # alpha - beta != 1