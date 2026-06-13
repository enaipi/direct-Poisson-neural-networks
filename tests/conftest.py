"""Pytest configuration and shared fixtures for DPNN tests."""

import pytest
import torch
import numpy as np


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device (CPU or GPU) for testing."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype():
    """Get the default dtype for testing."""
    return torch.float64


@pytest.fixture
def rb_params(device, dtype):
    """Standard parameters for RigidBody models."""
    return {
        "Ix": 1.0,
        "Iy": 1.0,
        "Iz": 1.0,
        "d2E": 1.0,
        "mx": 0.1,
        "my": 0.2,
        "mz": 0.3,
        "dt": 0.001,
        "alpha": 1.0,
        "T": 300,
        "device": device,
        "dtype": dtype,
    }


@pytest.fixture
def particle_3d_params(device, dtype):
    """Standard parameters for 3D Particle models."""
    return {
        "M": 1.0,
        "alpha": 1.0,
        "dt": 0.001,
        "init_rx": 0.1,
        "init_ry": 0.2,
        "init_rz": 0.3,
        "init_mx": 0.01,
        "init_my": 0.02,
        "init_mz": 0.03,
        "device": device,
        "dtype": dtype,
    }


@pytest.fixture
def particle_2d_params(device, dtype):
    """Standard parameters for 2D Particle models."""
    return {
        "M": 1.0,
        "alpha": 1.0,
        "zeta": 0.1,
        "dt": 0.001,
        "init_rx": 0.1,
        "init_ry": 0.2,
        "init_mx": 0.01,
        "init_my": 0.02,
        "device": device,
    }


@pytest.fixture
def heavy_top_params(device, dtype):
    """Standard parameters for HeavyTop models."""
    return {
        "Ix": 1.0,
        "Iy": 1.0,
        "Iz": 1.0,
        "d2E": 1.0,
        "mx": 0.1,
        "my": 0.2,
        "mz": 0.3,
        "init_rx": 0.0,
        "init_ry": 0.0,
        "init_rz": 1.0,
        "dt": 0.001,
        "alpha": 1.0,
        "Mgl": 1.0,
        "device": device,
    }


@pytest.fixture
def particle_nd_params(device, dtype):
    """Standard parameters for N-dimensional Particle models."""
    n_dims = 3
    return {
        "D": n_dims,
        "M": 1.0,
        "alpha": 1.0,
        "dt": 0.001,
        "init_r": torch.zeros(1, n_dims, dtype=dtype, device=device),
        "init_p": torch.zeros(1, n_dims, dtype=dtype, device=device),
        "B": 1,
        "device": device,
    }


@pytest.fixture
def shivamoggi_params(device, dtype):
    """Standard parameters for Shivamoggi models."""
    return {
        "M": 1.0,
        "dt": 0.001,
        "alpha": 1.0,
        "init_rx": 0.1,
        "init_ry": 0.2,
        "init_rz": 0.3,
        "init_u": 0.0,
        "device": device,
    }
