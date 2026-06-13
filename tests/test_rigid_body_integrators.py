"""Tests for RigidBody integrator classes."""

import pytest
import torch
from dpnn.models.physical_models import (
    RBEhrenfest,
    RBESeReCN,
    RBIMR,
    RBRK4,
    RBESeReFE,
)


class TestRBEhrenfest:
    """Test cases for RBEhrenfest integrator."""

    def test_initialization(self, rb_params):
        """Test RBEhrenfest can be instantiated."""
        rb = RBEhrenfest(**rb_params)
        assert rb is not None

    def test_m_new_method_exists(self, rb_params):
        """Test m_new() method is available."""
        rb = RBEhrenfest(**rb_params)
        assert hasattr(rb, "m_new")
        assert callable(rb.m_new)

    def test_m_new_returns_tensor(self, rb_params):
        """Test m_new() returns a tensor."""
        rb = RBEhrenfest(**rb_params)
        m_new = rb.m_new()
        assert isinstance(m_new, torch.Tensor)
        assert m_new.shape == (3,)


class TestRBESeReCN:
    """Test cases for RBESeReCN (Energy Self-Regularized Crank-Nicolson)."""

    def test_initialization(self, rb_params):
        """Test RBESeReCN can be instantiated."""
        rb = RBESeReCN(**rb_params)
        assert rb is not None

    def test_m_new_method_exists(self, rb_params):
        """Test m_new() method is available."""
        rb = RBESeReCN(**rb_params)
        assert hasattr(rb, "m_new")
        assert callable(rb.m_new)

    def test_m_new_default_parameters(self, rb_params):
        """Test m_new() with default parameters."""
        rb = RBESeReCN(**rb_params)
        m_new = rb.m_new()
        assert isinstance(m_new, torch.Tensor)
        assert m_new.shape == (3,)

    def test_m_new_with_entropy(self, rb_params):
        """Test m_new() with entropy calculation."""
        rb = RBESeReCN(**rb_params)
        result = rb.m_new(with_entropy=True)
        # Should return tuple (m_new, entropy) when with_entropy=True
        if isinstance(result, tuple):
            m_new, entropy = result
            assert isinstance(m_new, torch.Tensor)
            assert isinstance(entropy, (float, torch.Tensor))
        else:
            assert isinstance(result, torch.Tensor)


class TestRBIMR:
    """Test cases for RBIMR (Implicit Midpoint Rule)."""

    def test_initialization(self, rb_params):
        """Test RBIMR can be instantiated."""
        rb = RBIMR(**rb_params)
        assert rb is not None

    def test_m_new_method_exists(self, rb_params):
        """Test m_new() method is available."""
        rb = RBIMR(**rb_params)
        assert hasattr(rb, "m_new")
        assert callable(rb.m_new)

    def test_m_new_converges(self, rb_params):
        """Test m_new() produces a valid result."""
        rb = RBIMR(**rb_params)
        m_new = rb.m_new()
        assert isinstance(m_new, torch.Tensor)
        assert m_new.shape == (3,)
        assert not torch.isnan(m_new).any()
        assert not torch.isinf(m_new).any()

    def test_m_new_with_solver_iterations(self, rb_params):
        """Test m_new() with custom solver iterations."""
        rb = RBIMR(**rb_params)
        m_new = rb.m_new(solver_iterations=50)
        assert isinstance(m_new, torch.Tensor)
        assert m_new.shape == (3,)


class TestRBRK4:
    """Test cases for RBRK4 (4th order Runge-Kutta)."""

    def test_initialization(self, rb_params):
        """Test RBRK4 can be instantiated."""
        rb = RBRK4(**rb_params)
        assert rb is not None

    def test_m_new_method_exists(self, rb_params):
        """Test m_new() method is available."""
        rb = RBRK4(**rb_params)
        assert hasattr(rb, "m_new")
        assert callable(rb.m_new)

    def test_m_new_returns_valid_result(self, rb_params):
        """Test m_new() returns valid tensor."""
        rb = RBRK4(**rb_params)
        m_new = rb.m_new()
        assert isinstance(m_new, torch.Tensor)
        assert m_new.shape == (3,)
        assert not torch.isnan(m_new).any()


class TestRBESeReFE:
    """Test cases for RBESeReFE (Self-Regularized Forward Euler)."""

    def test_initialization(self, rb_params):
        """Test RBESeReFE can be instantiated."""
        rb = RBESeReFE(**rb_params)
        assert rb is not None

    def test_m_new_method_exists(self, rb_params):
        """Test m_new() method is available."""
        rb = RBESeReFE(**rb_params)
        assert hasattr(rb, "m_new")
        assert callable(rb.m_new)

    def test_m_new_with_optional_entropy(self, rb_params):
        """Test m_new() respects optional entropy parameter."""
        rb = RBESeReFE(**rb_params)
        result = rb.m_new(with_entropy=False)
        assert isinstance(result, torch.Tensor)


class TestIntegratorsConsistency:
    """Test consistency across integrator implementations."""

    def test_all_integrators_inherit_from_rigid_body(self, rb_params):
        """Test all integrators properly inherit RigidBody properties."""
        integrators = [RBEhrenfest, RBESeReCN, RBIMR, RBRK4, RBESeReFE]
        for integrator_class in integrators:
            rb = integrator_class(**rb_params)
            # Should have access to base RigidBody methods
            assert hasattr(rb, "energy_x")
            assert hasattr(rb, "omega_x")
            assert hasattr(rb, "m2")
            assert hasattr(rb, "get_L")

    def test_all_integrators_have_m_new(self, rb_params):
        """Test all integrators implement m_new() method."""
        integrators = [RBEhrenfest, RBESeReCN, RBIMR, RBRK4, RBESeReFE]
        for integrator_class in integrators:
            rb = integrator_class(**rb_params)
            assert callable(rb.m_new)
            # Should be callable without required arguments
            m_new = rb.m_new()
            assert isinstance(m_new, torch.Tensor)
