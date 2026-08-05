"""
Example: Analyzing Learned Hamiltonian Systems

Demonstrates how to use HamiltonianSystemAnalyzer to assess learning quality
for general Hamiltonian systems.

This shows:
1. Trajectory discrepancy analysis (learned vs ground truth)
2. Jacobi identity violation checking
3. Energy preservation analysis
4. Component-wise error breakdown
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.postprocessing import HamiltonianSystemAnalyzer
from dpnn.training import HamiltonianLearner
from dpnn.comparison import ComparisonConfig, ComparisonRunner


def analyze_rigid_body_example():
    """
    Full workflow: train a model and analyze results.
    """
    print("\n" + "="*70)
    print("ANALYSIS EXAMPLE: Rigid Body Dynamics")
    print("="*70 + "\n")
    
    # Step 1: Generate and train (using comparison runner)
    config = ComparisonConfig(
        model="RB",
        steps=200,              # Short for demo
        methods=["without", "soft"],
        neurons=32, layers=1,
        batch_size=32,
        epochs=3,              # Quick training
        generate=True,
        sampling=5,            # Few trajectories
        jacobi_loss_mode="spectral",
        folder_name="ANALYSIS_DEMO_RB",
        verbose=False,
    )
    
    print("Step 1: Training models...")
    runner = ComparisonRunner(config)
    runner.run()
    
    # Step 2: Load trained model and data
    print("\nStep 2: Loading trained model and data...")
    
    # Get the learner
    learner_soft = runner.learners["soft"]
    
    # Load training data
    import json
    data_path = Path("ANALYSIS_DEMO_RB/data/dataset.json")
    with open(data_path) as f:
        data = json.load(f)
    
    # Extract trajectories
    z_truth = np.array([traj["z"] for traj in data["trajectories"]])  # (num_traj, num_steps, dim)
    
    # Step 3: Generate predictions with learned model
    print("Step 3: Generating predictions with learned model...")
    
    z_learned_list = []
    for traj_idx, traj_data in enumerate(data["trajectories"]):
        z_traj = np.array(traj_data["z"])
        z_pred = [z_traj[0].copy()]
        
        # Forward simulation with learned dynamics
        dt = data["metadata"]["dt"]
        z_current = torch.tensor(z_traj[0], dtype=torch.float32)
        
        for step in range(len(z_traj) - 1):
            with torch.no_grad():
                # Compute ż = L(z) @ ∇H(z)
                z_current = z_current.clone().detach().requires_grad_(True)
                
                # Energy gradient
                E = learner_soft.energy(z_current.unsqueeze(0))
                E.backward()
                grad_H = z_current.grad.clone()
                
                # Structure matrix
                L_z = learner_soft.forward_L_tensor(z_current.unsqueeze(0))[0]  # (dim, dim)
                
                # Time step
                z_dot = L_z @ grad_H
                z_next = z_current + dt * z_dot
                z_pred.append(z_next.cpu().numpy())
                z_current = z_next
        
        z_learned_list.append(np.array(z_pred))
    
    z_learned = np.array(z_learned_list)
    
    # Step 4: Create analyzer and run analysis
    print("\nStep 4: Analyzing results...")
    analyzer = HamiltonianSystemAnalyzer(dimension=3, system_name="Rigid Body")
    
    # Trajectory discrepancy
    traj_results = analyzer.compute_trajectory_discrepancy(z_learned, z_truth, metric="rmse")
    analyzer.results["trajectory_discrepancy"] = traj_results
    print(f"  Mean trajectory error: {traj_results['mean_error']:.6e}")
    print(f"  Max trajectory error:  {traj_results['max_error']:.6e}")
    
    # Component errors
    comp_errors = analyzer.trajectory_error_per_component(z_learned, z_truth)
    analyzer.results["component_errors"] = comp_errors
    print(f"\n  Per-component errors:")
    for comp, err in comp_errors.items():
        print(f"    Component {comp}: {err:.6e}")
    
    # Jacobi identity error
    L_samples = []
    with torch.no_grad():
        for i in range(0, len(z_truth.flatten()) // 3, 10):  # Sample points
            idx = min(i, z_truth.shape[0]-1)
            step = min(i % z_truth.shape[1], z_truth.shape[1]-1)
            z_sample = torch.tensor(z_truth[idx, step], dtype=torch.float32)
            L_sample = learner_soft.forward_L_tensor(z_sample.unsqueeze(0))[0].numpy()
            L_samples.append(L_sample)
    
    L_samples = np.array(L_samples)
    jacobi_results = analyzer.compute_jacobi_error(L_samples, method="spectral")
    analyzer.results["jacobi_error"] = jacobi_results
    print(f"\n  Jacobi identity (antisymmetry) error:")
    print(f"    Mean: {jacobi_results['mean_antisymmetry_error']:.6e}")
    print(f"    Max:  {jacobi_results['max_antisymmetry_error']:.6e}")
    if "mean_eigenvalue_error" in jacobi_results:
        print(f"    Mean eigenvalue error: {jacobi_results['mean_eigenvalue_error']:.6e}")
    
    # Step 5: Generate visualizations
    print("\nStep 5: Generating visualizations...")
    
    # Trajectory comparison plot
    analyzer.plot_trajectory_discrepancy(
        z_learned, z_truth,
        trajectory_idx=0,
        component_indices=[0, 1, 2],
        save_path=Path("ANALYSIS_DEMO_RB/trajectory_comparison.png")
    )
    
    # Jacobi error histogram
    analyzer.plot_jacobi_error_histogram(
        jacobi_results["antisymmetry_error"],
        title="Jacobi Identity Violation (Antisymmetry Error)",
        save_path=Path("ANALYSIS_DEMO_RB/jacobi_error.png")
    )
    
    # Error evolution
    analyzer.plot_error_evolution(
        traj_results["step_errors"],
        title="Trajectory Error Evolution",
        save_path=Path("ANALYSIS_DEMO_RB/error_evolution.png")
    )
    
    # Step 6: Print report
    print("\n" + analyzer.generate_report())
    
    print("\nAnalysis complete! Results saved to ANALYSIS_DEMO_RB/")
    
    return analyzer, z_truth, z_learned


def analyze_from_saved_models(folder_name: str, method: str = "soft"):
    """
    Analyze results from a previous training run.
    
    Args:
        folder_name: Name of the folder containing results
        method: Which method to analyze ("without", "soft", "implicit")
    """
    print(f"\n{'='*70}")
    print(f"Analyzing saved results from {folder_name}")
    print(f"{'='*70}\n")
    
    folder = Path(folder_name)
    
    # Load data
    import json
    with open(folder / "data" / "dataset.json") as f:
        data = json.load(f)
    
    z_truth = np.array([traj["z"] for traj in data["trajectories"]])
    dim = data["metadata"]["dimension"]
    
    # Load trained model
    model_path = folder / "saved_models" / f"{method}_RB.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None
    
    print(f"Loading model from {model_path}")
    # This would load and initialize the learner
    # (Implementation depends on your model saving format)
    
    # Create analyzer
    analyzer = HamiltonianSystemAnalyzer(dimension=dim, system_name="Rigid Body (Saved)")
    
    # Print report
    print(analyzer.generate_report())
    
    return analyzer


# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# Hamiltonian System Analysis Examples")
    print("#"*70)
    
    # Run the full example
    analyzer, z_truth, z_learned = analyze_rigid_body_example()
    
    # Optional: analyze from saved models if they exist
    # analyzer = analyze_from_saved_models("ANALYSIS_DEMO_RB", method="soft")
    
    print("\nQuick access to analyzer:")
    print("  - analyzer.results['trajectory_discrepancy']")
    print("  - analyzer.results['jacobi_error']")
    print("  - analyzer.results['component_errors']")
    print("\nUse analyzer.plot_* methods for visualization.")
