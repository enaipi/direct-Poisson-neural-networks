"""
EXAMPLE 1: Learning Rigid Body Dynamics

Learn L(z) and H(z) for a rigid body (3D angular momentum).
Compares three methods: "without" (baseline), "implicit" (IMR), "soft" (Jacobi constraint).

This example uses the same infrastructure as comparison.py to avoid code duplication.
Uses ComparisonConfig and ComparisonRunner for consistent methodology.

System: 3D rigid body
  State z = (mx, my, mz) ∈ ℝ³ (angular momentum)
  Energy H(z) = 0.5 * m^T * I^-1 * m (kinetic energy)
  Structure L(z) = antisymmetric, learned from data
  
Expected: Soft method with spectral Jacobi should give best generalization.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import ComparisonConfig, ComparisonRunner
from dpnn.postprocessing import HamiltonianSystemAnalyzer


def main():
    """Run rigid body comparison using comparison.py infrastructure."""
    print("=" * 70)
    print("EXAMPLE 1: RIGID BODY DYNAMICS")
    print("=" * 70)
    
    # Create configuration for rigid body dynamics
    config = ComparisonConfig(
        # Model and simulation
        model="RB",                          # Rigid body
        steps=500,                           # Integration steps per trajectory
        
        # Methods to compare
        methods=["without", "implicit", "soft"],
        
        # Initial conditions (can adjust for different trajectories)
        init_mx=10.0,
        init_my=3.0,
        init_mz=4.0,
        
        # Model parameters (rigid body specific)
        Ix=10.0,     # Moment of inertia x
        Iy=20.0,     # Moment of inertia y
        Iz=40.0,     # Moment of inertia z
        
        # Learning parameters
        neurons=64,
        layers=2,
        batch_size=32,
        epochs=20,
        lr=0.001,
        dropout_rate=0.3,
        
        # Enable trajectory generation
        generate=True,
        sampling=10,      # Number of trajectories to generate
        seed=42,
        
        # Use spectral Jacobi constraint
        jacobi_loss_mode="spectral",
        
        # Output folder
        folder_name="examples_rigid_body",
        
        # Verbosity
        verbose=False
    )
    
    # Create and run comparison
    print("\nInitializing comparison runner...")
    runner = ComparisonRunner(config)
    
    print("Running full comparison...")
    runner.run()
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {config.folder_name}/")
    print(f"  - Training data: {config.folder_name}/data/")
    print(f"  - Trained models: {config.folder_name}/saved_models/")
    print(f"  - Configuration: {config.folder_name}/config.txt")
    
    # Analyze results with HamiltonianSystemAnalyzer
    analyze_results(runner, config)


def analyze_results(runner, config):
    """
    Analyze quality of learned Hamiltonian systems using HamiltonianSystemAnalyzer.
    """
    print(f"\n{'='*70}")
    print("POSTPROCESSING ANALYSIS")
    print(f"{'='*70}")
    
    # Load training data
    data_path = Path(config.folder_name) / "data" / "dataset.json"
    with open(data_path) as f:
        data = json.load(f)
    
    z_truth = np.array([traj["z"] for traj in data["trajectories"]])  # (num_traj, num_steps, dim)
    dt = data["metadata"]["dt"]
    dim = z_truth.shape[-1]
    
    # Analyze each method
    for method in config.methods:
        if method not in runner.learners:
            continue
            
        print(f"\n--- Method: {method} ---")
        learner = runner.learners[method]
        analyzer = HamiltonianSystemAnalyzer(dimension=dim, system_name=f"RigidBody_{method}")
        
        # Generate predictions
        z_pred_list = []
        for traj_data in data["trajectories"]:
            z_traj = np.array(traj_data["z"])
            z_pred = [z_traj[0].copy()]
            z_current = torch.tensor(z_traj[0], dtype=torch.float32)
            
            learner.energy.eval()
            learner.L_tensor.eval()
            
            with torch.no_grad():
                for step in range(len(z_traj) - 1):
                    z_current = z_current.clone().detach().requires_grad_(True)
                    H = learner.energy(z_current.unsqueeze(0))
                    H.backward()
                    grad_H = z_current.grad.clone()
                    
                    L_z = learner.forward_L_tensor(z_current.unsqueeze(0))[0]
                    z_dot = L_z @ grad_H
                    z_current = z_current.detach() + dt * z_dot
                    z_pred.append(z_current.detach().numpy())
            
            z_pred_list.append(np.array(z_pred))
        
        z_pred = np.array(z_pred_list)
        
        # Trajectory discrepancy
        traj_results = analyzer.compute_trajectory_discrepancy(z_pred, z_truth, metric="rmse")
        analyzer.results["trajectory_discrepancy"] = traj_results
        print(f"  Trajectory RMSE: mean={traj_results['mean_error']:.6e}, max={traj_results['max_error']:.6e}")
        
        # Jacobi error
        L_samples = []
        with torch.no_grad():
            for i in range(0, min(50, len(z_truth.flat)), max(1, len(z_truth.flat) // 50)):
                traj_idx = min(i // len(z_truth[0]), len(z_truth) - 1)
                step_idx = min(i % len(z_truth[0]), len(z_truth[0]) - 1)
                z_sample = torch.tensor(z_truth[traj_idx, step_idx], dtype=torch.float32)
                L_sample = learner.forward_L_tensor(z_sample.unsqueeze(0))[0].numpy()
                L_samples.append(L_sample)
        
        L_samples = np.array(L_samples)
        jacobi_results = analyzer.compute_jacobi_error(L_samples, method="spectral")
        analyzer.results["jacobi_error"] = jacobi_results
        print(f"  Jacobi error: antisymmetry={jacobi_results['mean_antisymmetry_error']:.6e}")


if __name__ == "__main__":
    main()
