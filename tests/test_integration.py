"""Integration tests for physical_models package."""

import pytest
import torch
from dpnn.models.physical_models import (
    RigidBody,
    RBIMR,
    Particle3DIMR,
    HeavyTopCN,
    Particle2DIMR,
    ShivamoggiIMR,
    ParticleNDCN,
)


class TestImportConsistency:
    """Test that all imports work from the public API."""

    def test_all_imports_from_physical_models(self):
        """Test all classes can be imported from physical_models."""
        from dpnn.models.physical_models import (
            RigidBody,
            load_models,
            RBEhrenfest,
            RBESeReCN,
            RBIMR,
            RBRK4,
            RBESeReFE,
            Neural,
            RBNeuralIMR,
            HeavyTopCN,
            HeavyTopIMR,
            HeavyTopNeural,
            HeavyTopNeuralIMR,
            Particle3DCN,
            Particle3DIMR,
            Particle3DNeural,
            Particle3DNeuralIMR,
            Particle3DKeplerIMR,
            Particle2DIMR,
            Particle2DNeural,
            ShivamoggiIMR,
            ShivamoggiNeural,
            ParticleNDCN,
            ParticleNDCNNeural,
        )
        # If we got here, all imports succeeded
        assert RigidBody is not None

    def test_all_imports_from_rigid_body_compat(self):
        """Test backward compatibility import path."""
        from dpnn.models.rigid_body import (
            RigidBody,
            load_models,
            RBIMR,
            HeavyTopCN,
            Particle3DIMR,
        )
        assert RigidBody is not None

    def test_wildcard_import(self):
        """Test wildcard import works correctly."""
        import dpnn.models.physical_models as pm

        # Key classes should be available
        assert hasattr(pm, "RigidBody")
        assert hasattr(pm, "load_models")


class TestModelInstantiationChain:
    """Test that models can be instantiated in sequence."""

    def test_create_multiple_models_sequentially(self, device, dtype):
        """Test creating multiple model instances in sequence."""
        # Create several different model types
        device = torch.device("cpu")

        imr = RBIMR(
            Ix=1.0,
            Iy=1.0,
            Iz=1.0,
            d2E=1.0,
            mx=0.1,
            my=0.2,
            mz=0.3,
            dt=0.001,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )
        assert imr is not None

        p3d = Particle3DIMR(
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_mx=0.01,
            init_my=0.02,
            init_mz=0.03,
            device=device,
            dtype=dtype,
        )
        assert p3d is not None

    def test_create_heavy_top_and_particle2d(self, device, dtype):
        """Test creating HeavyTop and Particle2D models."""
        ht = HeavyTopCN(
            Ix=1.0,
            Iy=1.0,
            Iz=1.0,
            d2E=1.0,
            mx=0.1,
            my=0.2,
            mz=0.3,
            init_rx=0.0,
            init_ry=0.0,
            init_rz=1.0,
            dt=0.001,
            alpha=1.0,
            Mgl=1.0,
            device=device,
        )
        assert ht is not None

        p2d = Particle2DIMR(
            M=1.0,
            alpha=1.0,
            zeta=0.1,
            dt=0.001,
            init_rx=0.1,
            init_ry=0.2,
            init_mx=0.01,
            init_my=0.02,
            device=device,
        )
        assert p2d is not None


class TestIntegration:
    """Integration tests across model systems."""

    def test_rigid_body_step(self, device, dtype):
        """Test a single step of RigidBody integration."""
        rb = RBIMR(
            Ix=1.0,
            Iy=1.0,
            Iz=1.0,
            d2E=1.0,
            mx=0.1,
            my=0.2,
            mz=0.3,
            dt=0.001,
            alpha=1.0,
            device=device,
            dtype=dtype,
        )
        m_new = rb.m_new()
        assert torch.isfinite(m_new).all()

    def test_particle_3d_step(self, device, dtype):
        """Test a single step of Particle3D integration."""
        p3d = Particle3DIMR(
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_rx=0.1,
            init_ry=0.2,
            init_rz=0.3,
            init_mx=0.01,
            init_my=0.02,
            init_mz=0.03,
            device=device,
            dtype=dtype,
        )
        r_new, p_new = p3d.m_new()
        assert torch.isfinite(r_new).all()
        assert torch.isfinite(p_new).all()

    def test_heavy_top_step(self, device, dtype):
        """Test a single step of HeavyTop integration."""
        ht = HeavyTopCN(
            Ix=1.0,
            Iy=1.0,
            Iz=1.0,
            d2E=1.0,
            mx=0.1,
            my=0.2,
            mz=0.3,
            init_rx=0.0,
            init_ry=0.0,
            init_rz=1.0,
            dt=0.001,
            alpha=1.0,
            Mgl=1.0,
            device=device,
        )
        m_new, r_new = ht.m_new()
        assert torch.isfinite(m_new).all()
        assert torch.isfinite(r_new).all()

    def test_shivamoggi_step(self, device, dtype):
        """Test a single step of Shivamoggi integration."""
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

    def test_particle_nd_step(self, device, dtype):
        """Test a single step of ParticleND integration."""
        p_nd = ParticleNDCN(
            D=3,
            M=1.0,
            alpha=1.0,
            dt=0.001,
            init_r=torch.zeros(1, 3, dtype=dtype, device=device),
            init_p=torch.zeros(1, 3, dtype=dtype, device=device),
            B=1,
            device=device,
        )
        q_new, p_new = p_nd.m_new()
        assert torch.isfinite(q_new).all()
        assert torch.isfinite(p_new).all()
