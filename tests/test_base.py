"""Tests for base RigidBody class and load_models utility."""

import pytest
import torch
from dpnn.models.physical_models import RigidBody, load_models


class TestRigidBody:
    """Test cases for RigidBody base class."""

    def test_rigid_body_initialization(self, rb_params):
        """Test RigidBody can be instantiated with valid parameters."""
        rb = RigidBody(**rb_params)
        assert rb is not None
        assert rb.Ix == 1.0
        assert rb.Iy == 1.0
        assert rb.Iz == 1.0
        assert rb.dt == 0.001

    def test_rigid_body_device_placement(self, rb_params):
        """Test RigidBody respects device parameter."""
        rb = RigidBody(**rb_params)
        assert rb.device == rb_params["device"]

    def test_rigid_body_dtype(self, rb_params):
        """Test RigidBody respects dtype parameter."""
        rb = RigidBody(**rb_params)
        assert rb.dtype == rb_params["dtype"]

    def test_energy_x_method(self, rb_params):
        """Test energy_x() method returns a scalar."""
        rb = RigidBody(**rb_params)
        energy = rb.energy_x()
        assert isinstance(energy, (float, torch.Tensor))

    def test_energy_y_method(self, rb_params):
        """Test energy_y() method returns a scalar."""
        rb = RigidBody(**rb_params)
        energy = rb.energy_y()
        assert isinstance(energy, (float, torch.Tensor))

    def test_energy_z_method(self, rb_params):
        """Test energy_z() method returns a scalar."""
        rb = RigidBody(**rb_params)
        energy = rb.energy_z()
        assert isinstance(energy, (float, torch.Tensor))

    def test_omega_x_method(self, rb_params):
        """Test omega_x() method returns a scalar."""
        rb = RigidBody(**rb_params)
        omega = rb.omega_x()
        assert isinstance(omega, (float, torch.Tensor))

    def test_omega_y_method(self, rb_params):
        """Test omega_y() method returns a scalar."""
        rb = RigidBody(**rb_params)
        omega = rb.omega_y()
        assert isinstance(omega, (float, torch.Tensor))

    def test_omega_z_method(self, rb_params):
        """Test omega_z() method returns a scalar."""
        rb = RigidBody(**rb_params)
        omega = rb.omega_z()
        assert isinstance(omega, (float, torch.Tensor))

    def test_m2_method(self, rb_params):
        """Test m2() method returns m^2."""
        rb = RigidBody(**rb_params)
        m_squared = rb.m2()
        assert isinstance(m_squared, (float, torch.Tensor))
        # m^2 should be positive
        if isinstance(m_squared, torch.Tensor):
            assert m_squared >= 0

    def test_m_magnitude_method(self, rb_params):
        """Test m_magnitude() method returns ||m||."""
        rb = RigidBody(**rb_params)
        m_mag = rb.m_magnitude()
        assert isinstance(m_mag, (float, torch.Tensor))
        # magnitude should be positive
        if isinstance(m_mag, torch.Tensor):
            assert m_mag >= 0

    def test_get_L_method(self, rb_params):
        """Test get_L() returns proper Poisson bivector tensor."""
        rb = RigidBody(**rb_params)
        m = torch.tensor([rb.mx, rb.my, rb.mz], dtype=rb.dtype, device=rb.device)
        L = rb.get_L(m)
        assert L.shape == (3, 3)
        # Poisson bivector should be skew-symmetric
        assert torch.allclose(L, -L.T)

    def test_get_E_method(self, rb_params):
        """Test get_E() returns proper energy structure tensor."""
        rb = RigidBody(**rb_params)
        m = torch.tensor([rb.mx, rb.my, rb.mz], dtype=rb.dtype, device=rb.device)
        E = rb.get_E(m)
        assert E.shape == (3, 3)
        # Energy structure should be symmetric
        assert torch.allclose(E, E.T)


class TestLoadModels:
    """Test cases for load_models utility function."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available, skipping GPU tests",
    )
    def test_load_models_gpu_device(self):
        """Test load_models with GPU device (if available)."""
        device = torch.device("cuda")
        try:
            result = load_models("test_model", "soft", "mx", device)
            assert result is not None
            # Should return tuple of (energy_net, L_net, J_net, A) or similar
            assert isinstance(result, tuple)
        except FileNotFoundError:
            # Expected if test_model doesn't exist
            pass

    def test_load_models_cpu_device(self):
        """Test load_models with CPU device."""
        device = torch.device("cpu")
        # This may fail if model files don't exist, but tests the interface
        try:
            result = load_models("nonexistent_model", "soft", "mx", device)
        except FileNotFoundError:
            # Expected behavior - model doesn't exist
            pass

    def test_load_models_different_methods(self):
        """Test load_models accepts different method types."""
        device = torch.device("cpu")
        methods = ["soft", "without", "implicit"]
        for method in methods:
            try:
                result = load_models("test", method, "mx", device)
            except FileNotFoundError:
                # Expected if models don't exist
                pass


class TestBackwardCompatibility:
    """Test backward compatibility with old rigid_body import path."""

    def test_import_from_rigid_body_compat(self):
        """Test that old import path still works."""
        from dpnn.models.rigid_body import RigidBody as RB_compat
        from dpnn.models.physical_models import RigidBody as RB_new

        # Should be the same class
        assert RB_compat is RB_new

    def test_wildcard_import_both_paths(self):
        """Test wildcard import works from both paths."""
        # Import from new path - store result to test
        import dpnn.models.physical_models as pm_new

        # Verify key classes are available
        assert hasattr(pm_new, "RigidBody")
        assert hasattr(pm_new, "load_models")

        # Import from old path
        import dpnn.models.rigid_body as rb_compat

        # Should still work
        assert hasattr(rb_compat, "RigidBody")
        assert hasattr(rb_compat, "load_models")
