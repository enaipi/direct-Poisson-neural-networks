"""
QUICK START: Recover L and H from a Time Series

This is a minimal, practical guide to:
1. Load your time series data
2. Train a learner using the same rigid-body training path as example 01
3. Extract the learned L(z) and H(z)
4. Analyze learning quality with HamiltonianSystemAnalyzer

Run this file as:
    python 00_quickstart.py
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    check_folder,
    generate_trajectories,
    resolve_automatic_dt,
)
from dpnn.training import DEFAULT_dataset

# ============================================================================
# STEP 1: PREPARE YOUR DATA
# ============================================================================

num_trajectories = 100
num_steps = 100  # Longer trajectories for more data

def rigid_body_generator_config():
    """Create the rigid-body config used by example 01."""
    return ComparisonConfig(
        model="RB",
        steps=num_steps,
        methods=["soft"],
        init_mx=10.0,
        init_my=3.0,
        init_mz=4.0,
        Ix=10.0,
        Iy=20.0,
        Iz=40.0,
        neurons=64,
        layers=2,
        batch_size=32,
        epochs=10,
        lr=0.001,
        dropout_rate=0.3,
        generate=True,
        sampling=num_trajectories,
        points=num_trajectories,
        seed=42,
        jacobi_loss_mode="spectral",
        folder_name="/tmp/quickstart_rigid_body",
        verbose=False,
    )


def reconstruct_rigid_body_trajectories(data_frame, steps_per_trajectory):
    """
    Convert the rigid-body generator's old/new transition rows into trajectories.

    The comparison generator writes one row per transition:
    (old_mx, old_my, old_mz) -> (mx, my, mz).
    This quickstart works with full z(t) arrays, so each trajectory is rebuilt as
    [old state from first row, then every new state].
    """
    rows_per_trajectory = steps_per_trajectory - 1
    if len(data_frame) % rows_per_trajectory != 0:
        raise ValueError(
            f"Expected generated row count to be divisible by {rows_per_trajectory}, "
            f"got {len(data_frame)}"
        )

    trajectories = []
    old_cols = ["old_mx", "old_my", "old_mz"]
    new_cols = ["mx", "my", "mz"]

    for start in range(0, len(data_frame), rows_per_trajectory):
        chunk = data_frame.iloc[start:start + rows_per_trajectory]
        initial_state = chunk.iloc[0][old_cols].to_numpy(dtype=np.float32)
        evolved_states = chunk[new_cols].to_numpy(dtype=np.float32)
        trajectories.append(np.vstack([initial_state, evolved_states]))

    return trajectories


def prepare_data():
    """
    Prepare Hamiltonian time series data for learning.
    
    Reuses the rigid-body trajectory generator from example 01:
    - State: z = (mx, my, mz), the angular momentum vector
    - Hamiltonian: H(z) = 0.5(mx^2/Ix + my^2/Iy + mz^2/Iz)
    - Dynamics: dz/dt = L(z) @ ∇H where L(z) is the rigid-body
      Lie-Poisson structure
    """

    dim = 3  # 3D: mx, my, mz
    config = rigid_body_generator_config()
    args = config.to_namespace()
    args.methods = []
    args.implicit = False
    args.soft = False
    args.without = False

    if args.dt <= 0.0:
        args.dt = resolve_automatic_dt(args)
        config.dt = args.dt

    check_folder(config.folder_name)
    generate_trajectories(args)

    generated_path = Path(config.folder_name) / DEFAULT_dataset
    generated_data = pd.read_csv(generated_path, dtype=np.float32)
    trajectories = reconstruct_rigid_body_trajectories(generated_data, num_steps)
    z_timeseries = np.concatenate(trajectories, axis=0)

    print(f"Generated rigid-body Hamiltonian trajectories:")
    print(f"  Shape: {z_timeseries.shape}")
    print(f"  States per trajectory: {num_steps}")
    print(f"  Number of trajectories: {len(trajectories)}")
    print(f"  State dimension: {dim} (mx, my, mz)")
    print(f"  Time step: {config.dt}")
    print(f"  System: Rigid Body (same generator as example 01)")
    print(f"  Generated data source: {generated_path}")

    return z_timeseries, dim, config.dt


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
    # (Each trajectory has num_steps states)
    num_steps_per_traj = len(z_timeseries) // num_trajectories
    trajectories = [
        z_timeseries[i*num_steps_per_traj:(i+1)*num_steps_per_traj]
        for i in range(num_trajectories)
    ]
    
    # Create standard JSON structure
    data = {
        "metadata": {
            "system_name": "RigidBody",
            "dimension": dim,
            "dt": dt,
            "num_trajectories": len(trajectories),
            "units": {"state": "angular momentum", "time": "s"}
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

def train_model(config, method="soft", epochs=None):
    """
    Train a learner to recover L and H using example 01's training path.
    
    Args:
        config: Rigid-body comparison config pointing at generated data
        method: "without" (baseline), "soft" (with Jacobi), "implicit" (IMR only)
        epochs: Optional override for the number of training epochs
    
    Returns:
        learner: Trained HamiltonianLearner object
    """
    if epochs is not None:
        config.epochs = epochs

    if config.dt <= 0.0:
        args = config.to_namespace()
        config.dt = resolve_automatic_dt(args)

    config.methods = [method]
    runner = ComparisonRunner(config)
    learner = runner._get_learner(method)

    print(f"\nTraining learner with example 01 approach:")
    print(f"  Model: {config.model}")
    print(f"  Data: {config.folder_name}/{DEFAULT_dataset}")
    print(f"  Method: {method}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Network: {config.neurons} neurons × {config.layers} layers")
    print(f"  Learning rate: {config.lr}")
    print(f"  Dropout: {config.dropout_rate}")

    learner.learn(
        method=method,
        learning_rate=config.lr,
        epochs=config.epochs,
        prefactor=config.prefactor,
        jac_prefactor=config.jac_prefactor
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
        system_name="RigidBody"
    )
    
    # -----------------------------------------------------------------------
    # Split concatenated trajectories back into individual trajectories
    # -----------------------------------------------------------------------
    num_steps_per_traj = len(z_timeseries) // num_trajectories
    z_traj_list = [
        z_timeseries[i*num_steps_per_traj:(i+1)*num_steps_per_traj]
        for i in range(num_trajectories)
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
    
    # STEP 2: Convert to standard format for users who want portable JSON data.
    # Training below intentionally uses the generated .xyz file, matching example 01.
    _data_path = convert_to_standard_format(z_timeseries, dim, dt)
    
    # STEP 3: Train learner with the same approach as examples/01_rigid_body.py
    training_config = rigid_body_generator_config()
    training_config.dt = dt
    learner = train_model(
        training_config,
        method="soft",      # Use soft Jacobi constraint
        epochs=10           # Same default as example 01
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
