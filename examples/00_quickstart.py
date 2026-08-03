"""
QUICK START: Recover L and H from a Time Series

This is a minimal, practical guide to:
1. Load your time series data
2. Train a learner
3. Extract the learned L(z) and H(z)

Run this file as:
    python recover_L_H.py
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
# STEP 5: COMPARE WITH GROUND TRUTH (if available)
# ============================================================================

def compare_with_ground_truth(learner, dim):
    """
    Optional: Compare learned dynamics with ground truth.
    
    If you know the true system, you can validate the learned L and H.
    """
    
    print("\n" + "="*70)
    print("VALIDATION (comparing learned vs true dynamics)")
    print("="*70)
    
    print("""
    If you have ground truth trajectories:
    
    1. Simulate with learned model:
        # Use energy and L_tensor networks to simulate
    
    2. Compare error:
        error = mean(|z_learned - z_true|)
    
    3. Check structure conservation:
        # Energy should be conserved
        H_initial = energy(z0)
        H_final = energy(z_final)
        energy_error = abs(H_final - H_initial)
        
        # Poisson structure should satisfy Jacobi identity
        # (checked automatically during training with Jacobi loss)
    
    4. Convergence metrics:
        # See validation losses during training
        # Implicit method: final validation error < 0.1 is good
        """)


# ============================================================================
# MAIN: Put it all together
# ============================================================================

def main():
    """
    Complete workflow: Load data → Train → Extract L and H
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
    
    # STEP 5: Validation
    compare_with_ground_truth(learner, dim)
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("""
    Next steps:
    
    1. Load your own data in STEP 1
    2. Adjust epochs, batch_size, neurons for better accuracy
    3. Try different methods: "without", "soft", "implicit"
    4. Export learned models for use in external code
    5. Compare learned dynamics with ground truth trajectories
    
    For more details, see:
    - L_H_RECOVERY_ARCHITECTURE.md (detailed explanation)
    - examples_general_learning.py (more examples)
    - GENERAL_LEARNING_ARCHITECTURE.md (system design)
    """)


if __name__ == "__main__":
    main()
