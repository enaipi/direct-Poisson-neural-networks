"""
EXAMPLE 2: Learning Heavy Top Dynamics

Learn L(z) and H(z) for a heavy top (spinning top with gravity).
Compares three methods: "without", "soft" (implicit not yet implemented for HT).

This example uses the same infrastructure as comparison.py to avoid code duplication.
Uses ComparisonConfig and ComparisonRunner for consistent methodology.

System: 6D heavy top
  State z = (rx, ry, rz, mx, my, mz) - orientation + angular momentum
  Energy H(z) = 0.5 * m^T * I^-1 * m + Mgl * rz
  Structure L(z) = antisymmetric, learned from data
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
    """Run heavy top comparison using comparison.py infrastructure."""
    print("=" * 70)
    print("EXAMPLE 2: HEAVY TOP DYNAMICS")
    print("=" * 70)
    
    # Create configuration for heavy top
    config = ComparisonConfig(
        # Model and simulation
        model="HT",                         # Heavy top
        steps=500,                          # Integration steps per trajectory
        
        # Methods to compare
        methods=["without", "soft"],        # Note: implicit not yet implemented for HT
        
        # Initial conditions
        init_mx=5.0,
        init_my=3.0,
        init_mz=2.0,
        init_rx=1.0,
        init_ry=0.0,
        init_rz=0.0,
        
        # Model parameters (heavy top specific)
        Ix=1.0,               # Moment of inertia x
        Iy=1.5,               # Moment of inertia y
        Iz=2.0,               # Moment of inertia z
        Mgl=9.81 * 0.1,       # Mass * gravity * distance
        
        # Learning parameters
        neurons=64,
        layers=2,
        batch_size=32,
        epochs=20,
        lr=0.001,
        dropout_rate=0.3,
        
        # Enable trajectory generation
        generate=True,
        sampling=12,          # Number of trajectories to generate
        seed=42,
        
        # Use spectral Jacobi constraint
        jacobi_loss_mode="spectral",
        
        # Output folder
        folder_name=str(RESULTS_DIR / "examples_heavy_top"),
        
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
        show_plots=False  # Don't block on matplotlib
    )
    
    return results


if __name__ == "__main__":
    main()
