"""
EXAMPLE 6: High-Dimensional Hamiltonian Learning (100D)

Learn L(z) and H(z) for a spring chain system with 50 beads (100D).
This example tests the scalability of DPNNs to high-dimensional systems.

System: 100D Spring Chain
  State: z = (q1, q2, ..., q50, p1, p2, ..., p50)
         qi = position of bead i
         pi = momentum of bead i
  
  Hamiltonian: H(z) = Σ pi²/2 + Σ (qi+1 - qi)²/2 + (q1²/2 + q50²/2)
  
  This is a canonical Hamiltonian system with clear structure.
  Learning the Poisson structure L should recover the symplectic form.

Run this file as:
    python 06_high_dimensional_example.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import ComparisonConfig, ComparisonRunner


def main():
    """High-dimensional spring chain learning."""
    print("=" * 70)
    print("EXAMPLE 6: LEARNING 100D SPRING CHAIN HAMILTONIAN")
    print("=" * 70)
    
    # Create configuration for 100D spring chain
    config = ComparisonConfig(
        # Model and simulation
        model="Sh",                         # Spring chain (harmonic)
        steps=500,                          # Steps per trajectory
        
        # Methods to compare
        methods=["without", "soft"],        # "implicit" is memory-intensive for 100D
        
        # Learning parameters (adjusted for high-dimensional system)
        neurons=256,                        # Larger network for high-dim
        layers=3,                           # More layers for capacity
        batch_size=8,                       # Small batches for memory efficiency
        epochs=30,                          # Sufficient for convergence
        lr=5e-4,                            # Moderate learning rate
        dropout_rate=0.1,                   # Some regularization
        
        # Generate trajectories
        generate=True,
        sampling=8,                         # 8 trajectories
        seed=42,
        
        # Use spectral Jacobi constraint
        jacobi_loss_mode="spectral",
        
        # Output folder
        folder_name="examples_100d_spring",
        
        # Verbosity
        verbose=False
    )
    
    # Create and run comparison
    print("\nInitializing comparison runner for 100D spring chain...")
    runner = ComparisonRunner(config)
    
    print("Running full comparison (this may take a few minutes)...")
    runner.run()
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {config.folder_name}/")
    print(f"  - Training data: {config.folder_name}/data/")
    print(f"  - Trained models: {config.folder_name}/saved_models/")
    print(f"  - Configuration: {config.folder_name}/config.txt")
    
    # Analyze results using standard postprocessing
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
