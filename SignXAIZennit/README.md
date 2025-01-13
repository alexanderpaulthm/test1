# SIGN-XAI PyTorch

PyTorch implementation of the SIGN (SIGNed explanations) method with Zennit integration for explainable AI.

## Installation

```bash
pip install signxai_torch

For development installation:

git clone https://github.com/your-username/signxai_torch.git
cd signxai_torch
pip install -e ".[dev]"

Usage
Basic usage example:

import torch
from signxai_torch.methods import SIGN

# Load your PyTorch model
model = ...  # Your PyTorch model
model.eval()

# Create SIGN analyzer
sign = SIGN(model)

# Prepare input
input_tensor = ...  # Your input tensor

# Get attributions
attributions = sign.attribute(input_tensor)

Using with specific target class:

# Get attributions for class 5
attributions = sign.attribute(input_tensor, target=5)

Using integrated gradients variant:

# Get integrated gradients attribution
attributions = sign.get_integrated_gradients(
    input_tensor,
    steps=50,
    baseline=torch.zeros_like(input_tensor)
)

Features
PyTorch-native implementation
Zennit integration for composite rules
Support for batch processing
Integrated Gradients variant
Comprehensive test suite
Compatible with modern PyTorch features
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

License
This project is licensed under the BSD 2-Clause License - see the LICENSE file for details.

Citation
If you use this software in your research, please cite:

@article{gumpfer2023sign,
    title={SIGN: Unveiling relevant features by reducing bias},
    author={Gumpfer, Nils},
    journal={...},
    year={2023}
}