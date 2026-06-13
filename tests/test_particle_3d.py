"""Tests for Particle3D model classes."""

import pytest
import torch
from dpnn.models.physical_models import (
    Particle3DCN,
    Particle3DIMR,
    Particle3DNeural,
    Particle3DNeuralIMR,
    Particle3DKeplerIMR,
)


class TestParticle3DCN:
    """Test cases for Particle3DCN (Crank-Nicolson 3D particle)."""

    def test_initialization(self, particle_3d_params):
        """Test Particle3DCN can be instantiated."""
        p = Particle3DCN(**particle_3d_params)
        assert p is not None
        assert p.M == 1.0
        assert p.alpha == 1.0

    def test_m_new_method_returns_tuple(self, particle_3d_params):
        """Test m_new() returns (r_new, p_new) tuple."""
        p = Particle3DCN(**particle_3d_params)
        result = p.m_new()
        assert isinstance(result, tuple)
        assert len(result) == 2
        r_new, p_new = result
        assert isinstance(r_new, torch.Tensor)
        assert isinstance(p_new, torch.Tensor)
        assert r_new.shape == (3,)
        assert p_new.shape == (3,)

    def test_position_momentum_dimensions(self, particle_3d_params):
        """Test that position and momentum are 3-dimensional."""
        p = Particle3DCN(**particle_3d_params)
        r_new, p_new = p.m_new()
        assert r_new.shape == (3,)
        assert p_new.shape == (3,)


class TestParticle3DIMR:
    """Test cases for Particle3DIMR (Implicit Midpoint Rule 3D particle)."""

    def test_initialization(self, particle_3d_params):
        """Test Particle3DIMR can be instantiated."""
        p = Particle3DIMR(**particle_3d_params)
        assert p is not None

    def test_m_new_method_returns_valid_tuple(self, particle_3d_params):
        """Test m_new() returns valid (r_new, p_new) tuple."""
        p = Particle3DIMR(**particle_3d_params)
        r_new, p_new = p.m_new()
        assert isinstance(r_new, torch.Tensor)
        assert isinstance(p_new, torch.Tensor)
        assert not torch.isnan(r_new).any()
        assert not torch.isnan(p_new).any()

    def test_convergence(self, particle_3d_params):
        """Test that IMR solver converges."""
        p = Particle3DIMR(**particle_3d_params)
        r_new, p_new = p.m_new(solver_iterations=100)
        # Should produce finite values
        assert torch.isfinite(r_new).all()
        assert torch.isfinite(p_new).all()


class TestParticle3DNeural:
    """Test cases for Particle3DNeural (neural network variant)."""

    def test_initialization(self, particle_3d_params):
        """Test Particle3DNeural can be instantiated."""
        try:
            p = Particle3DNeural(**particle_3d_params)
            assert p is not None
        except FileNotFoundError:
            # Expected if neural network files don't exist
            pytest.skip("Neural network model files not found")

    def test_m_new_method_exists(self, particle_3d_params):
        """Test m_new() method is available."""
        try:
            p = Particle3DNeural(**particle_3d_params)
            assert hasattr(p, "m_new")
            assert callable(p.m_new)
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestParticle3DNeuralIMR:
    """Test cases for Particle3DNeuralIMR (neural + IMR variant)."""

    def test_initialization(self, particle_3d_params):
        """Test Particle3DNeuralIMR can be instantiated."""
        try:
            p = Particle3DNeuralIMR(**particle_3d_params)
            assert p is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestParticle3DKeplerIMR:
    """Test cases for Particle3DKeplerIMR (Kepler problem with 1/r potential)."""

    def test_initialization(self, particle_3d_params):
        """Test Particle3DKeplerIMR can be instantiated."""
        p = Particle3DKeplerIMR(**particle_3d_params)
        assert p is not None

    def test_m_new_method_returns_tuple(self, particle_3d_params):
        """Test m_new() returns (r_new, p_new) tuple."""
        p = Particle3DKeplerIMR(**particle_3d_params)
        r_new, p_new = p.m_new()
        assert isinstance(r_new, torch.Tensor)
        assert isinstance(p_new, torch.Tensor)

    def test_kepler_potential_singularity_handling(self, particle_3d_params):
        """Test that Kepler singularity at origin is handled."""
        # Set position very close to origin to test singularity handling
        p_params = particle_3d_params.copy()
        p_params["rx"] = 1e-8
        p_params["ry"] = 1e-8
        p_params["rz"] = 1e-8
        p = Particle3DKeplerIMR(**p_params)
        r_new, p_new = p.m_new()
        # Should not produce NaN or Inf due to singularity handling
        assert torch.isfinite(r_new).all()
        assert torch.isfinite(p_new).all()


class TestParticle3DConsistency:
    """Test consistency across Particle3D implementations."""

    def test_all_variants_return_state_tuple(self, particle_3d_params):
        """Test all Particle3D variants return (r, p) state tuples."""
        variants = [Particle3DCN, Particle3DIMR, Particle3DKeplerIMR]
        for variant_class in variants:
            p = variant_class(**particle_3d_params)
            result = p.m_new()
            assert isinstance(result, tuple)
            assert len(result) == 2
            r_new, p_new = result
            assert r_new.shape == (3,) and p_new.shape == (3,)

    def test_all_variants_preserve_vector_types(self, particle_3d_params):
        """Test all variants preserve tensor types."""
        variants = [Particle3DCN, Particle3DIMR, Particle3DKeplerIMR]
        for variant_class in variants:
            p = variant_class(**particle_3d_params)
            r_new, p_new = p.m_new()
            assert isinstance(r_new, torch.Tensor)
            assert isinstance(p_new, torch.Tensor)
            assert r_new.dtype == particle_3d_params["dtype"]
            assert p_new.dtype == particle_3d_params["dtype"]
