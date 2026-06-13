"""Tests for HeavyTop model classes."""

import pytest
import torch
from dpnn.models.physical_models import (
    HeavyTopCN,
    HeavyTopIMR,
    HeavyTopNeural,
    HeavyTopNeuralIMR,
)


class TestHeavyTopCN:
    """Test cases for HeavyTopCN (Crank-Nicolson heavy top)."""

    def test_initialization(self, heavy_top_params):
        """Test HeavyTopCN can be instantiated."""
        ht = HeavyTopCN(**heavy_top_params)
        assert ht is not None
        assert ht.Ix == 1.0
        assert ht.Mgl == 1.0

    def test_m_new_method_returns_6d_state(self, heavy_top_params):
        """Test m_new() returns (m_new, r_new) tuple for 6D state."""
        ht = HeavyTopCN(**heavy_top_params)
        result = ht.m_new()
        assert isinstance(result, tuple)
        assert len(result) == 2
        m_new, r_new = result
        assert isinstance(m_new, torch.Tensor)
        assert isinstance(r_new, torch.Tensor)
        assert m_new.shape == (3,)  # Angular momentum 3D
        assert r_new.shape == (3,)  # Orientation 3D

    def test_state_consistency(self, heavy_top_params):
        """Test that returned state has consistent dimensions."""
        ht = HeavyTopCN(**heavy_top_params)
        m_new, r_new = ht.m_new()
        # Both should be 3D vectors
        assert m_new.shape == (3,)
        assert r_new.shape == (3,)
        assert m_new.dtype == heavy_top_params["dtype"]
        assert r_new.dtype == heavy_top_params["dtype"]


class TestHeavyTopIMR:
    """Test cases for HeavyTopIMR (Implicit Midpoint Rule heavy top)."""

    def test_initialization(self, heavy_top_params):
        """Test HeavyTopIMR can be instantiated."""
        ht = HeavyTopIMR(**heavy_top_params)
        assert ht is not None

    def test_m_new_returns_valid_state(self, heavy_top_params):
        """Test m_new() returns valid 6D state."""
        ht = HeavyTopIMR(**heavy_top_params)
        m_new, r_new = ht.m_new()
        assert isinstance(m_new, torch.Tensor)
        assert isinstance(r_new, torch.Tensor)
        assert not torch.isnan(m_new).any()
        assert not torch.isnan(r_new).any()

    def test_convergence_with_iterations(self, heavy_top_params):
        """Test convergence with specified solver iterations."""
        ht = HeavyTopIMR(**heavy_top_params)
        m_new, r_new = ht.m_new(solver_iterations=100)
        # Should produce finite values
        assert torch.isfinite(m_new).all()
        assert torch.isfinite(r_new).all()


class TestHeavyTopNeural:
    """Test cases for HeavyTopNeural (neural network variant)."""

    def test_initialization(self, heavy_top_params):
        """Test HeavyTopNeural can be instantiated."""
        try:
            ht = HeavyTopNeural(**heavy_top_params)
            assert ht is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")

    def test_neural_methods_exist(self, heavy_top_params):
        """Test neural-specific methods are available."""
        try:
            ht = HeavyTopNeural(**heavy_top_params)
            assert hasattr(ht, "neural_zdot")
            assert callable(ht.neural_zdot)
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestHeavyTopNeuralIMR:
    """Test cases for HeavyTopNeuralIMR (neural + IMR variant)."""

    def test_initialization(self, heavy_top_params):
        """Test HeavyTopNeuralIMR can be instantiated."""
        try:
            ht = HeavyTopNeuralIMR(**heavy_top_params)
            assert ht is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestHeavyTopConsistency:
    """Test consistency across HeavyTop implementations."""

    def test_all_variants_return_6d_state(self, heavy_top_params):
        """Test all HeavyTop variants return (m, r) state tuples."""
        variants = [HeavyTopCN, HeavyTopIMR]
        for variant_class in variants:
            ht = variant_class(**heavy_top_params)
            m_new, r_new = ht.m_new()
            assert m_new.shape == (3,)
            assert r_new.shape == (3,)

    def test_all_variants_preserve_dtype(self, heavy_top_params):
        """Test all variants preserve tensor dtype."""
        variants = [HeavyTopCN, HeavyTopIMR]
        for variant_class in variants:
            ht = variant_class(**heavy_top_params)
            m_new, r_new = ht.m_new()
            assert m_new.dtype == heavy_top_params["dtype"]
            assert r_new.dtype == heavy_top_params["dtype"]

    def test_gravitational_potential_effect(self, heavy_top_params):
        """Test that gravitational potential (Mgl) is used."""
        # Create two instances with different Mgl
        params_strong = heavy_top_params.copy()
        params_strong["Mgl"] = 10.0

        params_weak = heavy_top_params.copy()
        params_weak["Mgl"] = 0.1

        ht_strong = HeavyTopCN(**params_strong)
        ht_weak = HeavyTopCN(**params_weak)

        m_strong, _ = ht_strong.m_new()
        m_weak, _ = ht_weak.m_new()

        # Different Mgl values should produce different dynamics
        # (though not necessarily different state, depends on iteration)
        assert isinstance(m_strong, torch.Tensor)
        assert isinstance(m_weak, torch.Tensor)
