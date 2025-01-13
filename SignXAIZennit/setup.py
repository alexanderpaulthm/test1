"""Setup script for signxai_torch package."""

from setuptools import setup, find_packages

setup(
    name="signxai_torch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "zennit>=0.5.0",
        "numpy>=1.20.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "pillow>=8.0.0",
    ],
    extras_require={
       "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8",
    ],
    },
    author="Original: Nils Gumpfer, PyTorch Port: Community",
    author_email="nils.gumpfer@kite.thm.de",
    description="PyTorch implementation of SIGN-XAI with Zennit integration",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/signxai_torch",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)