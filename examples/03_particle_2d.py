"""
EXAMPLE 3: Learning 2D Particle Dynamics

Learn L(z) and H(z) for a classical 2D particle in an elliptic potential.
Compares three methods: "without", "implicit", "soft".

This example uses the same infrastructure as comparison.py to avoid code duplication.
Uses ComparisonConfig and ComparisonRunner for consistent methodology.

System: 4D canonical Hamiltonian particle
  State z = (x, y, px, py) - position + momentum
  Potential V(x,y) = x^2 + 2*y^2
  Energy H(z) = 0.5(px^2 + py^2) + x^2 + 2*y^2 (kinetic + potential)
  Structure L(z) = symplectic structure (linear for this system)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import ComparisonConfig, ComparisonRunner


def main():
    """Run particle 2D comparison using comparison.py infrastructure."""
    print("=" * 70)
    print("EXAMPLE 3: 2D PARTICLE IN ELLIPTIC POTENTIAL")
    print("=" * 70)
    
    # Create configuration for 2D particle
    config = ComparisonConfig(
        # Model and simulation
        model="P2D",                        # 2D particle
        steps=500,                          # Integration steps per trajectory
        
        # Methods to compare
        methods=["without", "soft"],        # Note: implicit not yet implemented for P2D
        
        # Initial conditions
        init_mx=0.5,                        # x momentum
        init_my=0.3,                        # y momentum
        init_rx=1.0,                        # x position
        init_ry=0.5,                        # y position
        
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
        folder_name="examples_particle_2d",
        
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


if __name__ == "__main__":
    main()
