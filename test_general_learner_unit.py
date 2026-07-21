"""
Unit Tests for GeneralSystemLearner and SystemSpec

Quick validation tests that don't require full training.
"""

import sys
import os
import torch
import numpy as np
import tempfile
import json

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dpnn import SystemSpec, GeneralSystemLearner, TrajectoryDataset
from dpnn.data import DatasetConverter, StandardDatasetLoader


def test_system_spec_registry():
    """Test SystemSpec registry."""
    print("\n[TEST] SystemSpec Registry")
    
    # Test predefined systems
    specs = [
        ('rigid_body', 3),
        ('heavy_top', 6),
        ('particle_3d', 6),
        ('particle_2d', 4),
    ]
    
    for name, expected_dim in specs:
        method = getattr(SystemSpec, name)
        spec = method()
        assert spec.dimension == expected_dim, \
            f"{name}: expected dim={expected_dim}, got {spec.dimension}"
        print(f"  ✓ {spec.name:20s} dim={spec.dimension}")
    
    # Test custom systems
    spec_custom = SystemSpec.custom("TestSystem", 7)
    assert spec_custom.dimension == 7
    print(f"  ✓ Custom system: TestSystem dim=7")
    
    return True


def test_system_spec_serialization():
    """Test SystemSpec to/from JSON."""
    print("\n[TEST] SystemSpec Serialization")
    
    spec = SystemSpec.rigid_body()
    
    # To dict
    spec_dict = spec.to_dict()
    assert spec_dict['name'] == 'RigidBody'
    assert spec_dict['dimension'] == 3
    print(f"  ✓ to_dict() works")
    
    # To JSON
    json_str = spec.to_json()
    assert 'RigidBody' in json_str
    print(f"  ✓ to_json() works")
    
    # From dict
    spec_restored = SystemSpec.from_dict(spec_dict)
    assert spec_restored.dimension == spec.dimension
    assert spec_restored.name == spec.name
    print(f"  ✓ from_dict() works")
    
    return True


def test_general_system_learner_creation():
    """Test GeneralSystemLearner instantiation."""
    print("\n[TEST] GeneralSystemLearner Creation")
    
    systems = [
        SystemSpec.rigid_body(),
        SystemSpec.heavy_top(),
        SystemSpec.particle_3d(),
        SystemSpec.particle_2d(),
    ]
    
    for spec in systems:
        learner = GeneralSystemLearner(
            spec,
            batch_size=16,
            neurons=32,
            layers=2,
            device='cpu'
        )
        
        # Verify networks created
        assert learner.energy_net is not None
        assert learner.jac_vec_net is not None
        assert learner.entropy_net is not None
        
        # Check dimension
        assert learner.dim == spec.dimension
        
        print(f"  ✓ {spec.name:20s} learner created")
    
    return True


def test_poisson_structures():
    """Test Poisson structure computation."""
    print("\n[TEST] Poisson Structure Computation")
    
    # Canonical (Particle3D)
    spec = SystemSpec.particle_3d()
    learner = GeneralSystemLearner(spec, device='cpu')
    
    z = torch.randn(4, 6)
    L = learner.get_poisson_structure(z)
    
    assert L.shape == (4, 6, 6)
    
    # Check antisymmetry: L^T = -L
    for i in range(4):
        antisym = L[i] + L[i].T
        assert torch.allclose(antisym, torch.zeros_like(antisym), atol=1e-6)
    
    print(f"  ✓ Canonical structure: antisymmetric ✓")
    
    # Rigid body
    spec = SystemSpec.rigid_body()
    learner = GeneralSystemLearner(spec, device='cpu')
    
    z = torch.randn(4, 3)
    L = learner.get_poisson_structure(z)
    
    assert L.shape == (4, 3, 3)
    
    # Check antisymmetry
    for i in range(4):
        antisym = L[i] + L[i].T
        assert torch.allclose(antisym, torch.zeros_like(antisym), atol=1e-6)
    
    print(f"  ✓ Rigid body structure: antisymmetric ✓")
    
    return True


def test_energy_gradient():
    """Test energy gradient computation."""
    print("\n[TEST] Energy Gradient Computation")
    
    spec = SystemSpec.rigid_body()
    learner = GeneralSystemLearner(spec, device='cpu')
    
    z = torch.randn(4, 3, requires_grad=True)
    
    # Compute energy
    E = learner.energy_net(z)
    
    # Manual backprop for validation
    E_sum = E.sum()
    E_sum.backward()
    
    grad_manual = z.grad.clone()
    z.grad = None
    
    # Using learner's method
    z2 = torch.randn(4, 3)
    grad_learner = learner.compute_energy_gradient(z2)
    
    # Both should have correct shape
    assert grad_manual.shape == (4, 3)
    assert grad_learner.shape == (4, 3)
    
    print(f"  ✓ Energy gradient shape: {grad_learner.shape}")
    print(f"  ✓ Gradient computation works")
    
    return True


def test_z_dot_prediction():
    """Test z_dot prediction."""
    print("\n[TEST] Dynamics Prediction (z_dot)")
    
    spec = SystemSpec.rigid_body()
    learner = GeneralSystemLearner(spec, device='cpu')
    
    z = torch.randn(4, 3)
    z_dot_pred = learner.compute_z_dot_pred(z)
    
    # Check shape and validity
    assert z_dot_pred.shape == (4, 3)
    assert not torch.isnan(z_dot_pred).any()
    assert not torch.isinf(z_dot_pred).any()
    
    print(f"  ✓ z_dot prediction shape: {z_dot_pred.shape}")
    print(f"  ✓ No NaN or Inf values")
    
    return True


def test_data_format_creation():
    """Test standard data format creation."""
    print("\n[TEST] Standard Data Format Creation")
    
    spec = SystemSpec.rigid_body()
    
    # Create synthetic trajectories
    trajectories = [np.random.randn(100, 3) for _ in range(5)]
    
    # Create JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        DatasetConverter.trajectories_to_standard_json(
            trajectories,
            spec,
            dt=0.01,
            json_path=json_path
        )
        
        # Validate
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data['metadata']['system_name'] == 'RigidBody'
        assert data['metadata']['dimension'] == 3
        assert data['metadata']['dt'] == 0.01
        assert len(data['trajectories']) == 5
        
        print(f"  ✓ JSON created with metadata")
        print(f"  ✓ {len(data['trajectories'])} trajectories included")
        print(f"  ✓ Dimension: {data['metadata']['dimension']}")
        
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)
    
    return True


def test_standard_dataset_loader():
    """Test loading standard format."""
    print("\n[TEST] Standard Dataset Loader")
    
    spec = SystemSpec.rigid_body()
    trajectories = [np.random.randn(100, 3) for _ in range(5)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # Create data
        DatasetConverter.trajectories_to_standard_json(
            trajectories, spec, dt=0.01, json_path=json_path
        )
        
        # Load metadata
        metadata, trajs = StandardDatasetLoader.load_json(json_path)
        
        assert metadata.system_name == 'RigidBody'
        assert metadata.dimension == 3
        assert len(trajs) == 5
        
        # Validate trajectories
        for traj in trajs:
            assert traj.z.shape[1] == 3
        
        print(f"  ✓ Metadata loaded: {metadata.system_name}")
        print(f"  ✓ {len(trajs)} trajectories loaded")
        print(f"  ✓ All trajectories validated")
        
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)
    
    return True


def test_trajectory_dataset_standard():
    """Test TrajectoryDataset.from_standard_json()."""
    print("\n[TEST] TrajectoryDataset Standard Format")
    
    spec = SystemSpec.rigid_body()
    trajectories = [np.random.randn(100, 3) * 2.0 for _ in range(5)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        # Create data
        DatasetConverter.trajectories_to_standard_json(
            trajectories, spec, dt=0.01, json_path=json_path
        )
        
        # Load with new API
        dataset = TrajectoryDataset.from_standard_json(json_path)
        
        # Validate
        assert len(dataset) > 0
        assert dataset.features.shape[1] == 3
        assert dataset.z_dot.shape == dataset.features.shape
        
        # Get sample
        z, z_dot, z_mid = dataset[0]
        assert z.shape == (3,)
        assert z_dot.shape == (3,)
        
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")
        print(f"  ✓ Feature shape: {dataset.features.shape}")
        print(f"  ✓ z_dot computed automatically")
        print(f"  ✓ Sample retrieval works")
        
    finally:
        if os.path.exists(json_path):
            os.remove(json_path)
    
    return True


def test_backward_compatibility():
    """Test backward compatibility with old API."""
    print("\n[TEST] Backward Compatibility (Old API)")
    
    from dpnn.training import create_learner
    
    # Old API should still work
    learner = create_learner("RB", batch_size=16)
    assert learner is not None
    assert learner.dim == 3
    print(f"  ✓ create_learner('RB') works")
    
    learner = create_learner("HT", batch_size=16)
    assert learner.dim == 6
    print(f"  ✓ create_learner('HT') works")
    
    learner = create_learner("P3D", batch_size=16)
    assert learner.dim == 6
    print(f"  ✓ create_learner('P3D') works")
    
    learner = create_learner("D", D=5, batch_size=16)
    assert learner.dim == 10
    print(f"  ✓ create_learner('D', D=5) works")
    
    return True


def test_loss_computation():
    """Test loss function computation."""
    print("\n[TEST] Loss Function Computation")
    
    spec = SystemSpec.rigid_body()
    learner = GeneralSystemLearner(spec, device='cpu')
    
    z = torch.randn(4, 3)
    z_dot_target = torch.randn(4, 3)
    
    # Compute loss
    loss = learner.loss_dynamics(z, z_dot_target)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)
    
    print(f"  ✓ Loss computed: {loss.item():.6f}")
    print(f"  ✓ Loss is positive and valid")
    
    return True


def test_multiple_systems():
    """Test that same code works for all systems."""
    print("\n[TEST] System Polymorphism")
    
    specs = [
        SystemSpec.rigid_body(),
        SystemSpec.heavy_top(),
        SystemSpec.particle_3d(),
        SystemSpec.particle_2d(),
    ]
    
    for spec in specs:
        # Create learner
        learner = GeneralSystemLearner(spec, batch_size=16, device='cpu')
        
        # Create dummy data
        z = torch.randn(4, spec.dimension)
        z_dot_target = torch.randn(4, spec.dimension)
        
        # Compute loss (generic code works for all!)
        loss = learner.loss_dynamics(z, z_dot_target)
        
        assert loss.item() > 0
        print(f"  ✓ {spec.name:20s}: loss={loss.item():.6f}")
    
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print(" "*20 + "UNIT TEST SUITE")
    print("="*70)
    
    tests = [
        test_system_spec_registry,
        test_system_spec_serialization,
        test_general_system_learner_creation,
        test_poisson_structures,
        test_energy_gradient,
        test_z_dot_prediction,
        test_data_format_creation,
        test_standard_dataset_loader,
        test_trajectory_dataset_standard,
        test_backward_compatibility,
        test_loss_computation,
        test_multiple_systems,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print(f"❌ {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
