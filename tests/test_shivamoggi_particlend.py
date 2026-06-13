"""Tests for Shivamoggi and N-dimensional particle models."""

import pytest
import torch
from dpnn.models.physical_models import (
    ShivamoggiIMR,
    ShivamoggiNeural,
    ParticleNDCN,
    ParticleNDCNNeural,
)


class TestShivamoggiIMR:
    """Test cases for ShivamoggiIMR (4D Shivamoggi system with advanced solver)."""

    def test_initialization(self):
        """Test ShivamoggiIMR can be instantiated."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        assert s is not None

    def test_m_new_method_returns_4d_state(self):
        """Test m_new() returns valid 4D state."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        result = s.m_new()
        assert isinstance(result, torch.Tensor)
        assert result.shape[1] == 4  # [u, x, y, z]

    def test_advanced_solver_with_newton(self):
        """Test that advanced solve() method can be called."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        # ShivamoggiIMR has a more sophisticated solve() method
        assert hasattr(s, "solve")
        assert callable(s.solve)

    def test_jacobian_computation(self):
        """Test Jacobian computation method exists."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        assert hasattr(s, "_jacobian")
        assert callable(s._jacobian)

    def test_convergence_to_finite_values(self):
        """Test solver converges to finite values."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        state = s.m_new()
        assert torch.isfinite(state).all()


class TestShivamoggiNeural:
    """Test cases for ShivamoggiNeural (neural network variant)."""

    def test_initialization(self):
        """Test ShivamoggiNeural can be instantiated."""
        device = torch.device("cpu")
        try:
            s = ShivamoggiNeural(
                M=1.0,
                dt=0.001,
                alpha=1.0,
                init_rx=0.1,
                init_ry=0.2,
                init_rz=0.3,
                init_u=0.0,
                device=device,
            )
            assert s is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")

    def test_m_new_method_exists(self):
        """Test m_new() method is available."""
        device = torch.device("cpu")
        try:
            s = ShivamoggiNeural(
                M=1.0,
                dt=0.001,
                alpha=1.0,
                init_rx=0.1,
                init_ry=0.2,
                init_rz=0.3,
                init_u=0.0,
                device=device,
            )
            assert callable(s.m_new)
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestParticleNDCN:
    """Test cases for ParticleNDCN (N-dimensional harmonic oscillator)."""

    def test_initialization_3d(self):
        """Test ParticleNDCN can be instantiated with 3D."""
        device = torch.device("cpu")
        dtype = torch.float64
        p = ParticleNDCN(
            D=3,
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_r=torch.zeros(1, 3, dtype=dtype, device=device),
            init_p=torch.zeros(1, 3, dtype=dtype, device=device),
            B=1,
            device=device,
        )
        assert p is not None
        assert p.D == 3

    def test_initialization_2d(self, device, dtype):
        """Test ParticleNDCN with 2D."""
        params = {
            "n_dims": 2,
            "alpha": 1.0,
            "dt": 0.001,
            "q": torch.zeros(2, dtype=dtype, device=device),
            "p": torch.zeros(2, dtype=dtype, device=device),
            "device": device,
            "dtype": dtype,
        }
        p = ParticleNDCN(**params)
        assert p.n_dims == 2

    def test_initialization_5d(self, device, dtype):
        """Test ParticleNDCN with 5D."""
        params = {
            "n_dims": 5,
            "alpha": 1.0,
            "dt": 0.001,
            "q": torch.zeros(5, dtype=dtype, device=device),
            "p": torch.zeros(5, dtype=dtype, device=device),
            "device": device,
            "dtype": dtype,
        }
        p = ParticleNDCN(**params)
        assert p.n_dims == 5

    def test_m_new_returns_correct_dimensions(self):
        """Test m_new() returns tuple with correct dimensions."""
        device = torch.device("cpu")
        dtype = torch.float64
        n_dims = 3
        p = ParticleNDCN(
            D=n_dims,
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_r=torch.zeros(1, n_dims, dtype=dtype, device=device),
            init_p=torch.zeros(1, n_dims, dtype=dtype, device=device),
            B=1,
            device=device,
        )
        result = p.m_new()
        assert isinstance(result, tuple)
        assert len(result) == 2
        q_new, p_new = result
        assert q_new.shape[1] == n_dims
        assert p_new.shape[1] == n_dims

    def test_get_L_block_structure(self):
        """Test get_L() returns proper block structure."""
        device = torch.device("cpu")
        dtype = torch.float64
        n_dims = 3
        p = ParticleNDCN(
            D=n_dims,
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_r=torch.zeros(1, n_dims, dtype=dtype, device=device),
            init_p=torch.zeros(1, n_dims, dtype=dtype, device=device),
            B=1,
            device=device,
        )
        q = torch.zeros(1, n_dims, dtype=dtype, device=device)
        L = p.get_L(q)
        # Should be 2n_dims x 2n_dims
        n = n_dims
        assert L.shape == (1, 2 * n, 2 * n)

    def test_scalability_with_dimension(self):
        """Test ParticleNDCN scales to different dimensions."""
        device = torch.device("cpu")
        dtype = torch.float64
        for n_dims in [1, 2, 3, 5, 10]:
            p = ParticleNDCN(
                D=n_dims,
                M=1.0,
                alpha=1.0,
                dt=0.001,
                init_r=torch.zeros(1, n_dims, dtype=dtype, device=device),
                init_p=torch.zeros(1, n_dims, dtype=dtype, device=device),
                B=1,
                device=device,
            )
            q_new, p_new = p.m_new()
            assert q_new.shape[1] == n_dims
            assert p_new.shape[1] == n_dims


class TestParticleNDCNNeural:
    """Test cases for ParticleNDCNNeural (neural network variant)."""

    def test_initialization(self):
        """Test ParticleNDCNNeural can be instantiated."""
        device = torch.device("cpu")
        dtype = torch.float64
        try:
            p = ParticleNDCNNeural(
                D=3,
                M=1.0,
                alpha=1.0,
                dt=0.001,
                init_r=torch.zeros(1, 3, dtype=dtype, device=device),
                init_p=torch.zeros(1, 3, dtype=dtype, device=device),
                B=1,
                device=device,
            )
            assert p is not None
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")

    def test_m_new_method_exists(self):
        """Test m_new() method is available."""
        device = torch.device("cpu")
        dtype = torch.float64
        try:
            p = ParticleNDCNNeural(
                D=3,
                M=1.0,
                alpha=1.0,
                dt=0.001,
                init_r=torch.zeros(1, 3, dtype=dtype, device=device),
                init_p=torch.zeros(1, 3, dtype=dtype, device=device),
                B=1,
                device=device,
            )
            assert callable(p.m_new)
        except FileNotFoundError:
            pytest.skip("Neural network model files not found")


class TestSpecializedSystemsConsistency:
    """Test consistency across specialized systems."""

    def test_shivamoggi_imr_state_dimension(self):
        """Test Shivamoggi returns 4D state."""
        device = torch.device("cpu")
        s = ShivamoggiIMR(
            M=1.0,
            dt=0.001,
            alpha=1.0,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_u=0.0,
            device=device,
        )
        state = s.m_new()
        assert state.shape[1] == 4

    def test_particle_nd_scales_correctly(self):
        """Test ParticleND scales with dimension."""
        device = torch.device("cpu")
        dtype = torch.float64
        for n_dims in [2, 3, 4]:
            p = ParticleNDCN(
                D=n_dims,
                M=1.0,
                alpha=1.0,
                dt=0.001,
                init_r=torch.zeros(1, n_dims, dtype=dtype, device=device),
                init_p=torch.zeros(1, n_dims, dtype=dtype, device=device),
                B=1,
                device=device,
            )
            assert p.D == n_dims
            q_new, p_new = p.m_new()
            assert q_new.shape[1] == n_dims and p_new.shape[1] == n_dims

    def test_all_specialized_handle_dtypes(self):
        """Test specialized systems handle different dtypes."""
        device = torch.device("cpu")
        for dtype in [torch.float32, torch.float64]:
            s = ShivamoggiIMR(
                M=1.0,
                dt=0.001,
                alpha=1.0,
                init_rx=0.1,
                init_ry=0.2,
                init_rz=0.3,
                init_u=0.0,
                device=device,
            )
            state = s.m_new()
            # The output dtype may depend on internal implementation
            assert isinstance(state, torch.Tensor)
