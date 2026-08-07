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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import ComparisonConfig, ComparisonRunner

# All example outputs (data/ and saved_models/) are collected under this
# top-level results folder, which is created automatically if missing.
RESULTS_DIR = Path("results")


def main():
    """Run rigid body comparison using comparison.py infrastructure."""
    print("=" * 70)
    print("EXAMPLE 1: RIGID BODY DYNAMICS")
    print("=" * 70)
    
    # Create configuration for rigid body dynamics
    config = ComparisonConfig(
        # Model and simulation
        model="RB",                          # Rigid body
        steps=100,                           # Integration steps per trajectory
        
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
        epochs=10,
        lr=0.001,
        dropout_rate=0.3,
        
        # Enable trajectory generation
        generate=True,
        sampling=10,      # Number of trajectories to generate
        seed=42,
        
        # Use spectral Jacobi constraint
        jacobi_loss_mode="spectral",
        
        # Output folder
        folder_name=str(RESULTS_DIR / "examples_rigid_body"),
        
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
    Analyze and plot training results using standard postprocessing pipeline.
    Shows trajectory fit errors and Jacobi identity errors.
    """
    from dpnn.postprocessing import run_postprocessing_analysis
    
    # Run the standard postprocessing analysis
    results = run_postprocessing_analysis(
        folder_name=config.folder_name,
        model=config.model,
        methods=config.methods,
        show_plots=False,  # Don't block on matplotlib
        verbose=True,      # Show the shared general-analysis report in addition to compatibility diagnostics
    )
    
    return results


if __name__ == "__main__":
    main()
