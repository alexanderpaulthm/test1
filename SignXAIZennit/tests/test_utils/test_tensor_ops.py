"""Tests for tensor operation utilities."""

import pytest
import torch
import numpy as np

from signxai_torch.utils.tensor_ops import (
    standardize_tensor,
    normalize_tensor,
    preprocess_image,
    deprocess_tensor
)

@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 32, 32)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

def test_standardize_tensor(sample_tensor):
    """Test tensor standardization."""
    standardized = standardize_tensor(sample_tensor)

    # Check mean and std
    assert torch.isclose(standardized.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(standardized.std(), torch.tensor(1.0), atol=1e-6)

    # Test with custom mean and std
    mean = 0.5
    std = 2.0
    standardized = standardize_tensor(sample_tensor, mean=mean, std=std)
    assert torch.isclose(
        (standardized * std + mean).mean(),
        sample_tensor.mean(),
        atol=1e-6
    )

def test_normalize_tensor(sample_tensor):
    """Test tensor normalization."""
    normalized = normalize_tensor(sample_tensor)

    # Check range [0, 1]
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)

    # Test with custom min and max
    min_val = -1
    max_val = 1
    normalized = normalize_tensor(sample_tensor, min_val=min_val, max_val=max_val)
    assert torch.isclose(normalized.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized.max(), torch.tensor(1.0), atol=1e-6)

def test_preprocess_image(sample_image):
    """Test image preprocessing."""
    # Test basic preprocessing
    tensor = preprocess_image(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4  # BCHW format
    assert tensor.shape[0] == 1  # Batch size 1
    assert tensor.shape[1] == 3  # 3 channels
    assert torch.all(tensor >= 0) and torch.all(tensor <= 1)

    # Test with normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = preprocess_image(sample_image, mean=mean, std=std)

    # Test single channel
    gray_image = sample_image.mean(axis=2, keepdims=True).astype(np.uint8)
    tensor = preprocess_image(gray_image)
    assert tensor.shape[1] == 1  # Single channel

    # Test tensor input
    tensor_input = torch.from_numpy(sample_image).float()
    tensor = preprocess_image(tensor_input)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4

def test_deprocess_tensor(sample_tensor):
    """Test tensor deprocessing."""
    # Normalize tensor to [0, 1]
    tensor = normalize_tensor(sample_tensor)

    # Test basic deprocessing
    deprocessed = deprocess_tensor(tensor)
    assert torch.all(deprocessed >= 0) and torch.all(deprocessed <= 1)

    # Test with mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized = preprocess_image(
        deprocess_tensor(tensor).numpy(),
        mean=mean,
        std=std
    )
    deprocessed = deprocess_tensor(normalized, mean=mean, std=std)
    assert torch.all(deprocessed >= 0) and torch.all(deprocessed <= 1)

def test_edge_cases():
    """Test edge cases for tensor operations."""
    # Empty tensor
    empty_tensor = torch.tensor([])
    with pytest.raises(RuntimeError):
        standardize_tensor(empty_tensor)

    # Zero std tensor
    zero_std = torch.ones(10, 10)
    standardized = standardize_tensor(zero_std)
    assert not torch.isnan(standardized).any()

    # Single value tensor
    single_value = torch.tensor([1.0])
    normalized = normalize_tensor(single_value)
    assert not torch.isnan(normalized).any()

    # Wrong dimension image
    wrong_dim = np.random.rand(32, 32, 4)  # 4 channels
    with pytest.raises(ValueError):
        preprocess_image(wrong_dim)

def test_device_handling(sample_tensor):
    """Test device handling in tensor operations."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cuda_tensor = sample_tensor.to(device)

        # Test standardization
        standardized = standardize_tensor(cuda_tensor)
        assert standardized.device == device

        # Test normalization
        normalized = normalize_tensor(cuda_tensor)
        assert normalized.device == device

        # Test deprocessing
        deprocessed = deprocess_tensor(cuda_tensor)
        assert deprocessed.device == device

def test_numerical_stability():
    """Test numerical stability of operations."""
    # Very small values
    small_tensor = torch.rand(10, 10) * 1e-10
    standardized = standardize_tensor(small_tensor)
    assert not torch.isnan(standardized).any()

    # Very large values
    large_tensor = torch.rand(10, 10) * 1e10
    normalized = normalize_tensor(large_tensor)
    assert not torch.isnan(normalized).any()

    # Mixed small and large values
    mixed_tensor = torch.cat([small_tensor, large_tensor])
    processed = normalize_tensor(mixed_tensor)
    assert not torch.isnan(processed).any()