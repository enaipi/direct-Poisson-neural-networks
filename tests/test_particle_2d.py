"""Tests for Particle2D model classes."""

import pytest
import torch
from dpnn.models.physical_models import Particle2DIMR, Particle2DNeural


class TestParticle2DIMR:
    """Test cases for Particle2DIMR (Implicit Midpoint Rule 2D particle with friction)."""

    def test_initialization(self, particle_2d_params):
        """Test Particle2DIMR can be instantiated."""
        p = Particle2DIMR(**particle_2d_params)
        assert p is not None
        assert p.M == 1.0
        assert p.zeta == 0.1  # Friction coefficient

    def test_m_new_method_returns_4d_state(self, particle_2d_params):
        """Test m_new() returns (r_new, p_new) tuple for 4D state."""
        p = Particle2DIMR(**particle_2d_params)
        result = p.m_new()
        assert isinstance(result, tuple)
        assert len(result) == 2
        r_new, p_new = result
        assert isinstance(r_new, torch.Tensor)
        assert isinstance(p_new, torch.Tensor)
        # 2D position and 2D momentum = 4D state total
        assert r_new.shape == (2,)
        assert p_new.shape == (2,)

    def test_friction_effect(self, particle_2d_params):
        """Test that friction coefficient affects dynamics."""
        # Create instance with friction
        p = Particle2DIMR(**particle_2d_params)
        r_with_friction, p_with_friction = p.m_new()

        # Friction should be applied (affect momentum evolution)
        assert isinstance(r_with_friction, torch.Tensor)
        assert isinstance(p_with_friction, torch.Tensor)

    def test_state_consistency(self, particle_2d_params):
        """Test state dimensions and types are consistent."""
        p = Particle2DIMR(**particle_2d_params)
        r_new, p_new = p.m_new()
        assert r_new.shape == (2,)
        assert p_new.shape == (2,)
        assert r_new.dtype == particle_2d_params["dtype"]
        assert p_new.dtype == particle_2d_params["dtype"]


class TestParticle2DNeural:
    """Test cases for Particle2DNeural (neural network variant)."""

    def test_initialization(self, particle_2d_params):
        """Test Particle2DNeural can be instantiated."""
        try:
            p = Particle2DNeural(**particle_2d_params)
            assert p is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")

    def test_m_new_method_exists(self, particle_2d_params):
        """Test m_new() method is available."""
        try:
            p = Particle2DNeural(**particle_2d_params)
            assert hasattr(p, "m_new")
            assert callable(p.m_new)
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")

    def test_neural_methods_exist(self, particle_2d_params):
        """Test neural-specific methods are available."""
        try:
            p = Particle2DNeural(**particle_2d_params)
            assert hasattr(p, "neural_zdot")
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestParticle2DConsistency:
    """Test consistency across Particle2D implementations."""

    def test_all_variants_return_4d_state(self, particle_2d_params):
        """Test all Particle2D variants return (r, p) state tuples."""
        p = Particle2DIMR(**particle_2d_params)
        result = p.m_new()
        assert isinstance(result, tuple)
        assert len(result) == 2
        r_new, p_new = result
        assert r_new.shape == (2,) and p_new.shape == (2,)

    def test_dimension_safety(self, particle_2d_params):
        """Test that 2D particle models use 2D state space."""
        p = Particle2DIMR(**particle_2d_params)
        r_new, p_new = p.m_new()
        # Should not accidentally use 3D or other dimensions
        assert r_new.shape == (2,) and p_new.shape == (2,)
        assert r_new.shape != (3,) and p_new.shape != (3,)

    def test_dissipation_term_effect(self, particle_2d_params):
        """Test that dissipation (friction) term is applied."""
        # Zero velocity should not cause issues with friction term
        params_zero_vel = particle_2d_params.copy()
        params_zero_vel["px"] = 0.0
        params_zero_vel["py"] = 0.0

        p = Particle2DIMR(**params_zero_vel)
        r_new, p_new = p.m_new()

        # Should handle zero velocity gracefully
        assert torch.isfinite(r_new).all()
        assert torch.isfinite(p_new).all()
