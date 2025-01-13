# tests/test_methods/test_wrapper.py

import pytest
import torch
import torch.nn as nn
from signxai_torch.methods.wrapper import calculate_relevancemap, validate_model_type, get_default_vgg_layer


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.mean(x, dim=[2, 3])
        return self.classifier(x)


@pytest.fixture
def setup_model_and_input():
    model = SimpleModel()
    inputs = torch.randn(1, 3, 32, 32)
    return model, inputs


def test_basic_gradient_methods(setup_model_and_input):
    model, inputs = setup_model_and_input

    # Test basic gradient methods
    methods = ["gradient", "input_t_gradient", "gradient_x_input"]
    for method in methods:
        result = calculate_relevancemap(method, inputs, model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_sign_methods(setup_model_and_input):
    model, inputs = setup_model_and_input

    # Test SIGN methods with different mu values
    methods = [
        "gradient_x_sign",
        "gradient_x_sign_mu_0",
        "gradient_x_sign_mu_0_5",
        "gradient_x_sign_mu_neg_0_5"
    ]
    for method in methods:
        result = calculate_relevancemap(method, inputs, model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_guided_methods(setup_model_and_input):
    model, inputs = setup_model_and_input

    methods = ["guided_backprop", "deconvnet"]
    for method in methods:
        result = calculate_relevancemap(method, inputs, model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_gradcam(setup_model_and_input):
    model, inputs = setup_model_and_input

    # Test GradCAM with explicit layer specification
    result = calculate_relevancemap("grad_cam", inputs, model, last_conv="features.2")
    assert isinstance(result, torch.Tensor)
    assert result.shape == inputs.shape

    # Test GradCAM without layer specification should raise error
    with pytest.raises(ValueError):
        calculate_relevancemap("grad_cam", inputs, model)


def test_integrated_gradients(setup_model_and_input):
    model, inputs = setup_model_and_input

    result = calculate_relevancemap(
        "integrated_gradients",
        inputs,
        model,
        steps=10,
        baseline=torch.zeros_like(inputs)
    )
    assert isinstance(result, torch.Tensor)
    assert result.shape == inputs.shape


def test_smoothgrad_methods(setup_model_and_input):
    model, inputs = setup_model_and_input

    methods = ["smoothgrad", "vargrad"]
    for method in methods:
        result = calculate_relevancemap(
            method,
            inputs,
            model,
            n_samples=5,
            noise_level=0.1
        )
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_lrp_methods(setup_model_and_input):
    model, inputs = setup_model_and_input

    # Test various LRP methods
    methods = [
        "lrp_z",
        "lrp_epsilon_0_1",
        "lrp_epsilon_1_std_x",
        "lrp_alpha_1_beta_0",
        "w2lrp_epsilon_0_1",
        "flatlrp_epsilon_0_1"
    ]
    for method in methods:
        result = calculate_relevancemap(method, inputs, model)
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_invalid_method():
    model = SimpleModel()
    inputs = torch.randn(1, 3, 32, 32)

    with pytest.raises(ValueError):
        calculate_relevancemap("invalid_method", inputs, model)


def test_helper_functions():
    model = SimpleModel()

    # Test validate_model_type
    assert validate_model_type(model, "Simple")
    assert not validate_model_type(model, "VGG")

    # Test get_default_vgg_layer
    with pytest.raises(ValueError):
        get_default_vgg_layer(model)


def test_target_specification(setup_model_and_input):
    model, inputs = setup_model_and_input

    # Test with different target specifications
    targets = [0, torch.tensor(0), torch.tensor([0])]
    for target in targets:
        result = calculate_relevancemap("gradient", inputs, model, target=target)
        assert isinstance(result, torch.Tensor)
        assert result.shape == inputs.shape


def test_batch_processing(setup_model_and_input):
    model, _ = setup_model_and_input
    batch_inputs = torch.randn(4, 3, 32, 32)

    result = calculate_relevancemap("gradient", batch_inputs, model)
    assert isinstance(result, torch.Tensor)
    assert result.shape == batch_inputs.shape