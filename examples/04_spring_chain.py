"""
EXAMPLE 4: Learning High-Dimensional Spring Chain Dynamics

Learn L(z) and H(z) for a coupled harmonic oscillator chain (200D).
Compares three methods: "without", "implicit", "soft".

This example uses the same infrastructure as comparison.py to avoid code duplication.
Uses ComparisonConfig and ComparisonRunner for consistent methodology.

System: 200D spring chain
  State z = (q1, q2, ..., q100, p1, p2, ..., p100)
  100 particles connected by springs
  Energy H(z) = Σ pi^2/2 + Σ(qi+1 - qi)^2/2
  Structure L(z) = antisymmetric, learned from data
  
Focus: Demonstrates scalability - spectral Jacobi applies to high-dimensional systems.
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import ComparisonConfig, ComparisonRunner


def main():
    """Run spring chain comparison using comparison.py infrastructure."""
    print("=" * 70)
    print("EXAMPLE 4: 200D SPRING CHAIN (100 Beads)")
    print("=" * 70)
    
    # Create configuration for spring chain
    config = ComparisonConfig(
        # Model and simulation
        model="Sh",                         # Spring chain (Harmonic)
        steps=300,                          # Shorter steps for high-dim (computational cost)
        
        # Methods to compare
        methods=["without", "soft"],        # Note: implicit not yet implemented for Sh
        
        # Learning parameters (adjusted for high-dimensional system)
        neurons=128,                        # Larger network for 200D
        layers=3,                           # More layers for complexity
        batch_size=16,                      # Smaller batches for memory efficiency
        epochs=20,
        lr=0.001,
        dropout_rate=0.3,
        
        # Enable trajectory generation
        generate=True,
        sampling=8,                         # Fewer trajectories (high-dimensional)
        seed=42,
        
        # Use spectral Jacobi constraint
        jacobi_loss_mode="spectral",
        
        # Output folder
        folder_name="examples_spring_chain",
        
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
