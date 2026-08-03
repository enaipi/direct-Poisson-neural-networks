"""
QUICK START: Recover L and H from a Time Series

This is a minimal, practical guide to:
1. Load your time series data
2. Train a learner
3. Extract the learned L(z) and H(z)
4. Analyze learning quality with HamiltonianSystemAnalyzer

Run this file as:
    python 00_quickstart.py
"""

import torch
import numpy as np
from pathlib import Path

# ============================================================================
# STEP 1: PREPARE YOUR DATA
# ============================================================================

def prepare_data():
    """
    Prepare time series data for learning.
    
    Input: Your time series as a numpy array or list
    Output: Standard JSON format that the learner can consume
    """
    
    # EXAMPLE: You have a time series of states
    # z(t_0), z(t_1), z(t_2), ..., z(t_n)
    
    # Option A: Load from your file
    # z_timeseries = np.load("your_trajectory.npy")  # shape: (n_steps, dim)
    # OR
    # import pandas as pd
    # df = pd.read_csv("your_data.csv")
    # z_timeseries = df[['z1', 'z2', 'z3', ...]].values  # shape: (n_steps, dim)
    
    # Option B: For this example, create synthetic data
    num_steps = 500
    dim = 3  # 3D system (e.g., rigid body)
    dt = 0.1
    
    # Synthetic trajectory: simple harmonic oscillator
    t = np.arange(num_steps) * dt
    z_timeseries = np.zeros((num_steps, dim))
    for i in range(num_steps):
        # Simple oscillation
        z_timeseries[i] = 5 * np.sin(t[i] * np.array([1.0, 1.5, 2.0]))
    
    print(f"Loaded time series: shape = {z_timeseries.shape}")
    print(f"  States per sample: {num_steps}")
    print(f"  State dimension: {dim}")
    print(f"  Time step: {dt}")
    
    return z_timeseries, dim, dt


# ============================================================================
# STEP 2: CONVERT TO STANDARD FORMAT
# ============================================================================

def convert_to_standard_format(z_timeseries, dim, dt):
    """
    Convert time series to standard JSON format.
    
    This creates training data pairs:
    (z_n, z_n+1) at times (t_n, t_n+1)
    """
    import json
    
    # Your single trajectory
    trajectories = [z_timeseries]  # Can have multiple trajectories
    
    # Create standard JSON structure
    data = {
        "metadata": {
            "system_name": "CustomSystem",
            "dimension": dim,
            "dt": dt,
            "num_trajectories": len(trajectories),
            "units": {"state": "unknown", "time": "s"}
        },
        "trajectories": [
            {
                "id": traj_id,
                "z": traj.tolist(),  # z(t_0), z(t_1), ...
                "t": (np.arange(len(traj)) * dt).tolist(),
                "metadata": {}
            }
            for traj_id, traj in enumerate(trajectories)
        ]
    }
    
    # Save to file
    output_path = Path("/tmp/my_data.json")
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"✓ Saved data to: {output_path}")
    return str(output_path)


# ============================================================================
# STEP 3: CREATE A LEARNER AND TRAIN
# ============================================================================

def train_model(data_path, dim, method="soft", epochs=20):
    """
    Train a learner to recover L and H from data using HamiltonianLearner.
    
    Args:
        data_path: Path to the standard JSON file
        dim: Dimension of the system
        method: "without" (baseline), "soft" (with Jacobi), "implicit" (IMR only)
        epochs: Number of training epochs
    
    Returns:
        learner: Trained HamiltonianLearner object
    """
    from dpnn.training.hamiltonian_learner import HamiltonianLearner
    from dpnn.system_spec import SystemSpec
    from dpnn.data.dataset import TrajectoryDataset
    
    print(f"\nTraining learner (method={method}, jacobi_mode=spectral, epochs={epochs})...")
    
    # Create generic system spec for this dimension to avoid legacy data loading
    system_spec = SystemSpec.custom(
        name=f"Generic_{dim}D",
        dimension=dim,
        description=f"Generic {dim}-dimensional system from standard JSON"
    )
    
    # Initialize learner with system spec (bypasses legacy data loading)
    device = torch.device('cpu')
    learner = HamiltonianLearner(
        system_spec=system_spec,
        batch_size=32,
        neurons=64,
        layers=2,
        device=device,
        dropout_rate=0.1,
        jacobi_loss_mode="spectral",  # Use spectral Jacobi constraint
        integration_scheme="imr",      # Implicit midpoint rule
        use_constant_L=False,           # Learn L(z) from data
        verbose=False
    )
    
    # Load data from standard JSON
    dataset = TrajectoryDataset.from_standard_json(data_path)
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    learner.train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True
    )
    learner.valid_loader = torch.utils.data.DataLoader(
        val_data, batch_size=32, shuffle=False
    )
    
    print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Train using HamiltonianLearner.learn() API
    learner.learn(
        method=method,
        learning_rate=1e-4,
        epochs=epochs,
        prefactor=1.0,
        jac_prefactor=0.01  # Weight for Jacobi constraint
    )
    
    print(f"✓ Training complete!")
    return learner


# ============================================================================
# STEP 4: EXTRACT AND USE LEARNED L(z) AND H(z)
# ============================================================================

def extract_learned_functions(learner, dim):
    """
    Extract the learned Poisson structure L(z) and energy H(z).
    
    Returns functions that can be called on new states.
    """
    
    print("\n" + "="*70)
    print("LEARNED FUNCTIONS")
    print("="*70)
    
    # Create test state
    z_test = torch.randn(1, dim)
    
    # -----------------------------------------------------------------------
    # 1. Energy H(z)
    # -----------------------------------------------------------------------
    H_z = learner.energy(z_test)
    print(f"\n1. Energy H(z)")
    print(f"   Input shape: {z_test.shape}")
    print(f"   Output shape: {H_z.shape}  (scalar)")
    print(f"   Example value: H({z_test[0].tolist()}) = {H_z.item():.6f}")
    
    # -----------------------------------------------------------------------
    # 2. Poisson Structure L(z)
    # -----------------------------------------------------------------------
    L_z = learner.forward_L_tensor(z_test)
    print(f"\n2. Poisson Structure L(z)")
    print(f"   Input shape: {z_test.shape}")
    print(f"   Output shape: {L_z.shape}  (antisymmetric matrix)")
    print(f"   Example matrix:")
    print(f"   {L_z[0].detach().numpy()}")
    print(f"   Antisymmetric? {torch.allclose(L_z + L_z.transpose(1, 2), torch.zeros_like(L_z))}")
    
    # -----------------------------------------------------------------------
    # 3. Energy Gradient ∇H(z)
    # -----------------------------------------------------------------------
    z_test_grad = torch.randn(1, dim, requires_grad=True)
    H = learner.energy(z_test_grad)
    grad_H = torch.autograd.grad(H.sum(), z_test_grad)[0]
    print(f"\n3. Energy Gradient ∇H(z)")
    print(f"   Shape: {grad_H.shape}")
    print(f"   Example: {grad_H[0].tolist()}")
    
    # -----------------------------------------------------------------------
    # 4. Dynamics ż = L(z) @ ∇H(z)
    # -----------------------------------------------------------------------
    z_dot = learner.compute_z_dot(z_test.detach())
    
    print(f"\n4. Learned Dynamics ż = L(z) @ ∇H(z)")
    print(f"   Shape: {z_dot.shape}")
    print(f"   Example velocity: {z_dot[0].tolist()}")
    
    # -----------------------------------------------------------------------
    # 5. Simulate forward in time
    # -----------------------------------------------------------------------
    print(f"\n5. Simulate Forward in Time")
    dt = 0.1
    z_current = torch.randn(1, dim)
    
    print(f"   t=0:    z = {z_current[0].tolist()}")
    
    learner.energy.eval()
    learner.L_tensor.eval()
    
    for t in range(1, 6):
        # Need gradients to compute ∇H even in eval mode
        z_current_req = z_current.clone().detach().requires_grad_(True)
        z_dot = learner.compute_z_dot(z_current_req)
        z_current = (z_current + dt * z_dot.detach()).detach()
        print(f"   t={t*dt:.1f}:  z = {z_current[0].tolist()}")
    
    # -----------------------------------------------------------------------
    # Summary: Access the learned functions
    # -----------------------------------------------------------------------
    print(f"\n" + "="*70)
    print("API SUMMARY: How to use L(z) and H(z)")
    print("="*70)
    print("""
    Once trained, the learner provides all functions:
    
    # Energy function
    H_value = learner.energy(z)  # Input: (batch, dim) → Output: (batch, 1)
    
    # Poisson structure
    L = learner.forward_L_tensor(z)  # Input: (batch, dim) → Output: (batch, dim, dim)
    
    # Dynamics ż = L(z) @ ∇H(z)
    z_dot = learner.compute_z_dot(z)  # Computes L(z) @ ∇H(z) automatically
    
    # Simulation
    z_next = z_current + dt * learner.compute_z_dot(z_current)
    
    # Export for external use
    # Save the neural networks:
    torch.save(learner.energy.state_dict(), "energy.pt")
    torch.save(learner.L_tensor.state_dict(), "L_tensor.pt")
    
    # Load in another script:
    from dpnn.models import EnergyNet, TensorNet
    energy = EnergyNet(dim, neurons=64, layers=2)
    energy.load_state_dict(torch.load("energy.pt"))
    """)
    
    return learner


# ============================================================================
# STEP 5: ANALYZE LEARNED MODEL QUALITY
# ============================================================================

def analyze_learned_model(learner, z_timeseries, dim, dt):
    """
    Use HamiltonianSystemAnalyzer to measure learning quality.
    
    Computes:
    - Trajectory discrepancy (learned vs ground truth)
    - Jacobi identity error (Poisson structure validation)
    - Energy conservation
    - Per-component error breakdown
    """
    from dpnn.postprocessing import HamiltonianSystemAnalyzer
    
    print("\n" + "="*70)
    print("STEP 5: ANALYZE LEARNED MODEL QUALITY")
    print("="*70)
    
    # Create analyzer
    analyzer = HamiltonianSystemAnalyzer(
        dimension=dim,
        system_name="QuickstartExample"
    )
    
    # -----------------------------------------------------------------------
    # Generate predictions with learned model
    # -----------------------------------------------------------------------
    print("\nGenerating predictions with learned model...")
    
    z_pred = [z_timeseries[0]]  # Start with first state
    
    learner.energy.eval()
    learner.L_tensor.eval()
    
    with torch.no_grad():
        z_current = torch.tensor(z_timeseries[0], dtype=torch.float32)
        
        for step in range(1, len(z_timeseries)):
            # Compute ∇H
            z_current = z_current.clone().detach().requires_grad_(True)
            H = learner.energy(z_current.unsqueeze(0))
            H.backward()
            grad_H = z_current.grad.clone()
            
            # Get L(z)
            with torch.no_grad():
                L_z = learner.forward_L_tensor(z_current.unsqueeze(0))[0]
                z_dot = L_z @ grad_H
                z_next = z_current.detach() + dt * z_dot
                z_pred.append(z_next.numpy())
                z_current = z_next
    
    z_pred = np.array(z_pred)
    print(f"  Generated {len(z_pred)} predictions")
    
    # -----------------------------------------------------------------------
    # Trajectory Discrepancy Analysis
    # -----------------------------------------------------------------------
    print("\nAnalyzing trajectory discrepancy...")
    
    traj_results = analyzer.compute_trajectory_discrepancy(
        z_pred[np.newaxis, :, :],  # Wrap as (1, num_steps, dim)
        z_timeseries[np.newaxis, :, :],
        metric="rmse"
    )
    
    analyzer.results["trajectory_discrepancy"] = traj_results
    
    print(f"  Mean RMSE:   {traj_results['mean_error']:.6e}")
    print(f"  Max RMSE:    {traj_results['max_error']:.6e}")
    print(f"  Median RMSE: {traj_results['median_error']:.6e}")
    
    # -----------------------------------------------------------------------
    # Jacobi Identity Analysis (Poisson Structure)
    # -----------------------------------------------------------------------
    print("\nAnalyzing Poisson structure (Jacobi identity)...")
    
    # Sample L matrices at different states
    L_samples = []
    num_samples = min(50, len(z_timeseries))
    sample_indices = np.linspace(0, len(z_timeseries)-1, num_samples, dtype=int)
    
    with torch.no_grad():
        for idx in sample_indices:
            z_sample = torch.tensor(z_timeseries[idx], dtype=torch.float32)
            L_sample = learner.forward_L_tensor(z_sample.unsqueeze(0))[0].numpy()
            L_samples.append(L_sample)
    
    L_samples = np.array(L_samples)
    
    jacobi_results = analyzer.compute_jacobi_error(L_samples, method="spectral")
    analyzer.results["jacobi_error"] = jacobi_results
    
    print(f"  Antisymmetry error:  {jacobi_results['mean_antisymmetry_error']:.6e}")
    print(f"    Max antisymmetry:  {jacobi_results['max_antisymmetry_error']:.6e}")
    if "mean_eigenvalue_error" in jacobi_results:
        print(f"  Eigenvalue error:    {jacobi_results['mean_eigenvalue_error']:.6e}")
    
    # -----------------------------------------------------------------------
    # Per-Component Error
    # -----------------------------------------------------------------------
    print("\nPer-component trajectory errors:")
    
    comp_errors = analyzer.trajectory_error_per_component(
        z_pred[np.newaxis, :, :],
        z_timeseries[np.newaxis, :, :]
    )
    analyzer.results["component_errors"] = comp_errors
    
    for comp_idx, error in comp_errors.items():
        print(f"  Component {comp_idx}: {error:.6e}")
    
    # -----------------------------------------------------------------------
    # Generate Report
    # -----------------------------------------------------------------------
    print("\n" + analyzer.generate_report())
    
    return analyzer


# ============================================================================
# STEP 6: COMPARE WITH GROUND TRUTH (if available)
# ============================================================================

def compare_with_ground_truth(learner, dim):
    """
    Optional: Compare learned dynamics with ground truth.
    
    If you know the true system, you can validate the learned L and H.
    """
    
    print("\n" + "="*70)
    print("STEP 6: FURTHER VALIDATION")
    print("="*70)
    
    print("""
    The HamiltonianSystemAnalyzer (STEP 5) provides comprehensive analysis:
    
    1. Trajectory Discrepancy:
        - Measures how well learned model reproduces data
        - Good range: RMSE < 1e-3
    
    2. Jacobi Identity (Poisson Structure):
        - Verifies L + L^T = 0 (antisymmetry)
        - Checks eigenvalues are purely imaginary
        - Good range: error < 1e-4
    
    3. Per-Component Errors:
        - Shows which variables are learned better
        - Balanced errors indicate good learning
    
    4. Energy Conservation:
        - Optional: compare energy along trajectory
        - Good range: relative error < 1%
    
    Advanced validation with ground truth:
    
    - Generate long-term predictions and compare
    - Check Lyapunov exponents (for chaotic systems)
    - Validate Casimir invariants if applicable
    - Use analyzer.plot_*() for visualizations
    """)


# ============================================================================
# MAIN: Put it all together
# ============================================================================

def main():
    """
    Complete workflow: Load data → Train → Extract L and H → Analyze
    """
    
    print("="*70)
    print("RECOVER L(z) AND H(z) FROM TIME SERIES")
    print("="*70)
    
    # STEP 1: Prepare data
    z_timeseries, dim, dt = prepare_data()
    
    # STEP 2: Convert to standard format
    data_path = convert_to_standard_format(z_timeseries, dim, dt)
    
    # STEP 3: Train learner
    learner = train_model(
        data_path, 
        dim, 
        method="soft",  # Use soft Jacobi constraint
        epochs=10       # Fewer epochs for quick demo
    )
    
    # STEP 4: Extract learned functions
    learned_funcs = extract_learned_functions(learner, dim)
    
    # STEP 5: Analyze learned model quality
    analyzer = analyze_learned_model(learner, z_timeseries, dim, dt)
    
    # STEP 6: Further validation
    compare_with_ground_truth(learner, dim)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("""
    Next steps:
    
    1. Load your own data in STEP 1
    2. Adjust epochs, batch_size, neurons for better accuracy
    3. Try different methods: "without", "soft", "implicit"
    4. Use analyzer.plot_*() to visualize results
    5. Export learned models for use in external code
    6. Compare learned dynamics with ground truth trajectories
    
    For more details, see:
    - L_H_RECOVERY_ARCHITECTURE.md (detailed explanation)
    - examples_general_learning.py (more examples)
    - GENERAL_LEARNING_ARCHITECTURE.md (system design)
    """)


if __name__ == "__main__":
    main()
