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

num_trajectories = 100
num_steps = 100  # Longer trajectories for more data

def prepare_data():
    """
    Prepare Hamiltonian time series data for learning.
    
    Generates trajectories from a 2D harmonic oscillator:
    - State: z = (q1, p1, q2, p2) where q=position, p=momentum
    - Hamiltonian: H(z) = 0.5(p1^2 + p2^2) + 0.5(q1^2 + 4*q2^2)
    - Dynamics: dz/dt = L @ ∇H where L is the symplectic structure
    
    This is a canonical Hamiltonian system that the learner can recover.
    """
    
    dim = 4  # 4D: q1, p1, q2, p2
    dt = 0.001  # Very small time step for numerical stability
    
    # Generate multiple trajectories with different initial conditions
    trajectories = []
    for traj_idx in range(num_trajectories):  # More trajectories
        # Initial conditions: energy-limited (not too large)
        q1_0 = 1.0 + 0.3 * traj_idx
        p1_0 = 0.8 + 0.2 * traj_idx
        q2_0 = 0.5 - 0.15 * traj_idx
        p2_0 = 0.6 + 0.1 * traj_idx
        
        z = np.zeros((num_steps, dim))
        z[0] = [q1_0, p1_0, q2_0, p2_0]
        
        # Integrate using symplectic Stormer-Verlet integrator (better structure preservation)
        # This is more stable and better for learning Hamiltonian structure
        for step in range(num_steps - 1):
            q1, p1, q2, p2 = z[step]
            
            # Half-step momentum update: p = p + 0.5*dt*(-∇V)
            # where ∇V = (q1, 0, 4*q2, 0) for V = 0.5*q1^2 + 0.5*4*q2^2
            p1 = p1 - 0.5 * dt * q1
            p2 = p2 - 0.5 * dt * 4 * q2
            
            # Full-step position update: q = q + dt*p (kinetic energy coefficient = 1)
            q1 = q1 + dt * p1
            q2 = q2 + dt * p2
            
            # Half-step momentum update
            p1 = p1 - 0.5 * dt * q1
            p2 = p2 - 0.5 * dt * 4 * q2
            
            z[step + 1] = [q1, p1, q2, p2]
        
        trajectories.append(z)
    
    # Concatenate all trajectories
    z_timeseries = np.concatenate(trajectories, axis=0)
    
    # Normalize to reasonable scale (helps with learning)
    z_mean = np.mean(z_timeseries, axis=0)
    z_std = np.std(z_timeseries, axis=0)
    z_std = np.where(z_std < 1e-6, 1.0, z_std)  # Avoid division by near-zero
    z_timeseries = (z_timeseries - z_mean) / z_std
    
    print(f"Generated Hamiltonian trajectories:")
    print(f"  Shape: {z_timeseries.shape}")
    print(f"  States per trajectory: {num_steps}")
    print(f"  Number of trajectories: {num_trajectories}")
    print(f"  State dimension: {dim} (q1, p1, q2, p2)")
    print(f"  Time step: {dt}")
    print(f"  System: 2D Harmonic Oscillator (canonical Hamiltonian)")
    print(f"  Data normalized - Mean: {z_mean}, Std: {z_std}")
    
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
    
    # Split concatenated trajectories back into individual trajectories
    # (Each trajectory has ~2000 steps)
    num_steps_per_traj = len(z_timeseries) // num_trajectories
    trajectories = [
        z_timeseries[i*num_steps_per_traj:(i+1)*num_steps_per_traj]
        for i in range(num_trajectories)
    ]
    
    # Create standard JSON structure
    data = {
        "metadata": {
            "system_name": "HarmonicOscillator2D",
            "dimension": dim,
            "dt": dt,
            "num_trajectories": len(trajectories),
            "units": {"state": "canonical (q,p)", "time": "s"}
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
    batch_size = 8  # Small batches for stable training
    
    learner = HamiltonianLearner(
        system_spec=system_spec,
        batch_size=batch_size,
        neurons=128,        # Larger network for better fitting
        layers=3,           # More layers for complexity
        device=device,
        dropout_rate=0.02,  # Very low dropout for small dataset
        jacobi_loss_mode="spectral",  # Use spectral Jacobi constraint
        integration_scheme="imr",      # Implicit midpoint rule
        use_constant_L=False,           # Learn L(z) from data
        verbose=False
    )
    
    # Load data from standard JSON
    dataset = TrajectoryDataset.from_standard_json(data_path)
    
    # Create data loaders with same batch size
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    learner.train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    learner.valid_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )
    
    print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Network: {128} neurons × 3 layers")
    
    # Train using HamiltonianLearner.learn() API
    learner.learn(
        method=method,
        learning_rate=1e-4,     # Very small learning rate for small normalized data
        epochs=epochs,
        prefactor=0.5,          # Scale main loss (focus on Jacobi)
        jac_prefactor=1.0       # High weight for Jacobi (critical for Hamiltonian structure)
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
    
    Computes and displays:
    - Trajectory discrepancy (learned vs ground truth)
    - Jacobi identity error (Poisson structure validation)
    - Per-component error breakdown
    """
    from dpnn.postprocessing import HamiltonianSystemAnalyzer
    
    print("\n" + "="*70)
    print("STEP 5: ANALYZE LEARNED MODEL QUALITY")
    print("="*70)
    
    # Create analyzer
    analyzer = HamiltonianSystemAnalyzer(
        dimension=dim,
        system_name="HarmonicOscillator2D"
    )
    
    # -----------------------------------------------------------------------
    # Split concatenated trajectories back into individual trajectories
    # -----------------------------------------------------------------------
    num_steps_per_traj = len(z_timeseries) // 4
    z_traj_list = [
        z_timeseries[i*num_steps_per_traj:(i+1)*num_steps_per_traj]
        for i in range(4)
    ]
    
    print(f"\nGenerating predictions for {len(z_traj_list)} trajectories...")
    
    learner.energy.eval()
    learner.L_tensor.eval()
    
    # Generate predictions for each trajectory
    z_pred_list = []
    
    for traj_idx, z_traj_truth in enumerate(z_traj_list):
        z_pred = [z_traj_truth[0]]  # Start with first state
        z_current = torch.tensor(z_traj_truth[0], dtype=torch.float32)
        
        for step in range(1, len(z_traj_truth)):
            # Compute ∇H (needs gradients)
            z_current = z_current.clone().detach().requires_grad_(True)
            H = learner.energy(z_current.unsqueeze(0))
            H.backward()
            grad_H = z_current.grad.detach().clone()  # Properly detach before using
            
            # Get L(z) and advance (no gradients needed)
            with torch.no_grad():
                L_z = learner.forward_L_tensor(z_current.unsqueeze(0))[0]
                z_dot = L_z @ grad_H  # Both tensors are now detached
                z_next = (z_current.detach() + dt * z_dot).cpu().numpy()
                z_pred.append(z_next)
                z_current = torch.tensor(z_next, dtype=torch.float32)
        
        z_pred_list.append(np.array(z_pred))
    
    z_pred_array = np.array(z_pred_list)
    z_truth_array = np.array(z_traj_list)
    
    print(f"✓ Generated {len(z_pred_list)} trajectories")
    print(f"  Shapes - Learned: {z_pred_array.shape}, Truth: {z_truth_array.shape}")
    
    # Check for NaN/inf values
    pred_has_nan = np.any(np.isnan(z_pred_array)) or np.any(np.isinf(z_pred_array))
    truth_has_nan = np.any(np.isnan(z_truth_array)) or np.any(np.isinf(z_truth_array))
    print(f"  Learned has NaN/inf: {pred_has_nan}")
    print(f"  Truth has NaN/inf: {truth_has_nan}")
    
    if pred_has_nan:
        print(f"  WARNING: Learned predictions contain NaN/inf!")
        nan_count = np.sum(np.isnan(z_pred_array))
        inf_count = np.sum(np.isinf(z_pred_array))
        print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
        print(f"    Sample learned values: {z_pred_array[0, :5, :]}")
    
    if truth_has_nan:
        print(f"  WARNING: Ground truth contains NaN/inf!")
    
    # Check for trajectory divergence/explosion
    pred_max = np.nanmax(np.abs(z_pred_array))
    truth_max = np.nanmax(np.abs(z_truth_array))
    print(f"  Max absolute values - Learned: {pred_max:.6e}, Truth: {truth_max:.6e}")
    
    if pred_max > 1e3:
        print(f"  WARNING: Learned trajectory has exploded! (max > 1e3)")
    
    # -----------------------------------------------------------------------
    # Trajectory Fit Error (matching unified pipeline)
    # -----------------------------------------------------------------------
    print("\n--- Trajectory Fit Error ---")
    
    try:
        # Compute difference
        diff = z_pred_array - z_truth_array
        rmse_per_point = np.sqrt(np.mean(diff**2, axis=2))  # (num_traj, num_steps)
        
        # Check for NaN
        if np.any(np.isnan(rmse_per_point)):
            print(f"  WARNING: RMSE computation resulted in NaN")
            print(f"    Diff has NaN: {np.any(np.isnan(diff))}")
            print(f"    Diff has inf: {np.any(np.isinf(diff))}")
            print(f"    Diff min: {np.nanmin(diff)}, max: {np.nanmax(diff)}")
        
        # Flatten to get all RMSE values
        rmse_flat = rmse_per_point.flatten()
        rmse_flat = rmse_flat[~np.isnan(rmse_flat)]  # Remove NaN for stats
        
        if len(rmse_flat) == 0:
            print(f"  ERROR: All RMSE values are NaN!")
            traj_results = {
                'mean_error': np.nan,
                'median_error': np.nan,
                'max_error': np.nan
            }
        else:
            traj_results = {
                'mean_error': np.mean(rmse_flat),
                'median_error': np.median(rmse_flat),
                'max_error': np.max(rmse_flat)
            }
        
        analyzer.results["trajectory_discrepancy"] = traj_results
        
        print(f"  Mean RMSE:   {traj_results['mean_error']:.6e}")
        print(f"  Median RMSE: {traj_results['median_error']:.6e}")
        print(f"  Max RMSE:    {traj_results['max_error']:.6e}")
    except Exception as e:
        print(f"  Error computing trajectory discrepancy: {e}")
        import traceback
        traceback.print_exc()
    
    # -----------------------------------------------------------------------
    # Jacobi Identity Error (matching unified pipeline)
    # -----------------------------------------------------------------------
    print("\n--- Jacobi Identity Error ---")
    
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
    
    try:
        jacobi_results = analyzer.compute_jacobi_error(L_samples, method="spectral")
        analyzer.results["jacobi_error"] = jacobi_results
        
        print(f"  Antisymmetry error:  {jacobi_results['mean_antisymmetry_error']:.6e}")
        if "max_antisymmetry_error" in jacobi_results:
            print(f"  Max antisymmetry:    {jacobi_results['max_antisymmetry_error']:.6e}")
    except Exception as e:
        print(f"  Error computing Jacobi error: {e}")
    
    # -----------------------------------------------------------------------
    # Per-Component Error (optional detail)
    # -----------------------------------------------------------------------
    print("\n--- Per-Component Error ---")
    
    try:
        comp_errors = analyzer.trajectory_error_per_component(
            z_pred_array,
            z_truth_array
        )
        analyzer.results["component_errors"] = comp_errors
        
        for comp_idx in sorted(comp_errors.keys()):
            print(f"  Component {comp_idx}: {comp_errors[comp_idx]:.6e}")
    except Exception as e:
        print(f"  Error computing per-component errors: {e}")
    
    # -----------------------------------------------------------------------
    # Summary Report
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
        method="soft",      # Use soft Jacobi constraint
        epochs=50           # More epochs for Hamiltonian data
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
