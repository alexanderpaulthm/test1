#!/usr/bin/env python3
"""Test runner script for signxai_torch."""

import sys
import pytest

if __name__ == "__main__":
    # Define default pytest arguments
    pytest_args = [
        "tests/",
        "-v",  # verbose output
        "--capture=no",  # show stdout/stderr
    ]

    # Add coverage args if pytest-cov is installed
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=signxai_torch",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    except ImportError:
        print("pytest-cov not installed. Running without coverage reporting.")

    # Run tests
    sys.exit(pytest.main(pytest_args))