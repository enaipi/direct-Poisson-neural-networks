"""Helpers for analyzing learned Hamiltonian models in a general postprocessing workflow."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from .hamiltonian_analysis import HamiltonianSystemAnalyzer


def _coerce_trajectory_list(z_timeseries, num_trajectories: int):
    """Normalize either concatenated time series or trajectory arrays into a list."""
    if isinstance(z_timeseries, (list, tuple)):
        return [np.asarray(traj, dtype=np.float32) for traj in z_timeseries]

    z_array = np.asarray(z_timeseries, dtype=np.float32)

    if z_array.ndim == 3:
        return [z_array[i] for i in range(z_array.shape[0])]

    if z_array.ndim == 2:
        if num_trajectories is None:
            raise ValueError("num_trajectories must be provided for concatenated time series")
        num_steps_per_traj = len(z_array) // num_trajectories
        if num_steps_per_traj * num_trajectories != len(z_array):
            raise ValueError(
                f"Expected concatenated trajectory length to be divisible by {num_trajectories}, "
                f"got {len(z_array)}"
            )
        return [
            z_array[i * num_steps_per_traj : (i + 1) * num_steps_per_traj]
            for i in range(num_trajectories)
        ]

    raise ValueError(f"Unsupported trajectory shape: {z_array.shape}")


def analyze_general_model_data(
    z_learned,
    z_truth,
    L_matrices,
    dimension: int,
    system_name: str = "General",
    analyzer: Optional[HamiltonianSystemAnalyzer] = None,
    verbose: bool = True,
    learner=None,
    state_samples=None,
):
    """Analyze learned trajectories and structure matrices directly from arrays."""
    if analyzer is None:
        analyzer = HamiltonianSystemAnalyzer(dimension=dimension, system_name=system_name)

    z_learned_array = np.asarray(z_learned, dtype=np.float32)
    z_truth_array = np.asarray(z_truth, dtype=np.float32)
    L_samples = np.asarray(L_matrices, dtype=np.float32)

    if verbose:
        print("\n" + "=" * 70)
        print("GENERAL MODEL ANALYSIS")
        print("=" * 70)

    try:
        traj_results = analyzer.compute_trajectory_discrepancy(z_learned_array, z_truth_array, metric="rmse")
        analyzer.results["trajectory_discrepancy"] = traj_results
        if verbose:
            print(f"  Mean RMSE:   {traj_results['mean_error']:.6e}")
            print(f"  Median RMSE: {traj_results['median_error']:.6e}")
            print(f"  Max RMSE:    {traj_results['max_error']:.6e}")
    except Exception as exc:  # pragma: no cover - defensive path
        if verbose:
            print(f"  Error computing trajectory discrepancy: {exc}")

    try:
        jacobi_loss_fn = getattr(learner, "jacobi_loss", None) if learner is not None else None
        jacobi_results = analyzer.compute_jacobi_error(
            L_samples,
            method="spectral",
            state_samples=state_samples,
            jacobi_loss_fn=jacobi_loss_fn,
        )
        analyzer.results["jacobi_error"] = jacobi_results
        if verbose:
            primary_jacobi_error = jacobi_results.get(
                "mean_spectral_jacobi_loss",
                jacobi_results.get("mean_jacobi_identity_error", np.nan),
            )
            print(f"  Jacobi identity error: {primary_jacobi_error:.6e}")
            print(f"  Mean kernel rank:      {jacobi_results['mean_kernel_rank']:.6e}")
            if "max_jacobi_identity_error" in jacobi_results:
                print(f"  Max Jacobi error:      {jacobi_results['max_jacobi_identity_error']:.6e}")
            if "spectral_jacobi_loss" in jacobi_results:
                print(f"  Spectral Jacobi loss:  {jacobi_results['spectral_jacobi_loss']:.6e}")
    except Exception as exc:  # pragma: no cover - defensive path
        if verbose:
            print(f"  Error computing Jacobi error: {exc}")

    try:
        comp_errors = analyzer.trajectory_error_per_component(z_learned_array, z_truth_array)
        analyzer.results["component_errors"] = comp_errors
        if verbose:
            for comp_idx in sorted(comp_errors.keys()):
                print(f"  Component {comp_idx}: {comp_errors[comp_idx]:.6e}")
    except Exception as exc:  # pragma: no cover - defensive path
        if verbose:
            print(f"  Error computing per-component errors: {exc}")

    if verbose:
        print("\n" + analyzer.generate_report())

    return analyzer


def analyze_general_model(
    learner,
    z_timeseries,
    dim: int,
    dt: float,
    num_trajectories: int,
    analyzer: Optional[HamiltonianSystemAnalyzer] = None,
    verbose: bool = True,
):
    """Analyze a learned model by rolling it forward over trajectories."""
    if analyzer is None:
        analyzer = HamiltonianSystemAnalyzer(dimension=dim, system_name="RigidBody")

    z_traj_list = _coerce_trajectory_list(z_timeseries, num_trajectories)

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: ANALYZE LEARNED MODEL QUALITY")
        print("=" * 70)
        print(f"\nGenerating predictions for {len(z_traj_list)} trajectories...")

    energy_module = getattr(learner, "energy", None)
    structure_module = getattr(learner, "L_tensor", None)

    if energy_module is not None:
        energy_module.eval()
    if structure_module is not None:
        structure_module.eval()

    z_pred_list = []
    for _, z_traj_truth in enumerate(z_traj_list):
        z_pred = [z_traj_truth[0]]
        z_current = torch.tensor(z_traj_truth[0], dtype=torch.float32)

        for step in range(1, len(z_traj_truth)):
            z_current = z_current.clone().detach().requires_grad_(True)
            H = energy_module(z_current.unsqueeze(0))
            H.backward()
            grad_H = z_current.grad.detach().clone()

            with torch.no_grad():
                L_z = learner.forward_L_tensor(z_current.unsqueeze(0))[0]
                z_dot = L_z @ grad_H
                z_next = (z_current.detach() + dt * z_dot).cpu().numpy()
                z_pred.append(z_next)
                z_current = torch.tensor(z_next, dtype=torch.float32)

        z_pred_list.append(np.array(z_pred))

    z_pred_array = np.array(z_pred_list)
    z_truth_array = np.array(z_traj_list)

    if verbose:
        print(f"✓ Generated {len(z_pred_list)} trajectories")
        print(f"  Shapes - Learned: {z_pred_array.shape}, Truth: {z_truth_array.shape}")

        pred_has_nan = np.any(np.isnan(z_pred_array)) or np.any(np.isinf(z_pred_array))
        truth_has_nan = np.any(np.isnan(z_truth_array)) or np.any(np.isinf(z_truth_array))
        print(f"  Learned has NaN/inf: {pred_has_nan}")
        print(f"  Truth has NaN/inf: {truth_has_nan}")

        if pred_has_nan:
            print("  WARNING: Learned predictions contain NaN/inf!")
            nan_count = np.sum(np.isnan(z_pred_array))
            inf_count = np.sum(np.isinf(z_pred_array))
            print(f"    NaN count: {nan_count}, Inf count: {inf_count}")
            print(f"    Sample learned values: {z_pred_array[0, :5, :]}")

        if truth_has_nan:
            print("  WARNING: Ground truth contains NaN/inf!")

        pred_max = np.nanmax(np.abs(z_pred_array))
        truth_max = np.nanmax(np.abs(z_truth_array))
        print(f"  Max absolute values - Learned: {pred_max:.6e}, Truth: {truth_max:.6e}")

        if pred_max > 1e3:
            print("  WARNING: Learned trajectory has exploded! (max > 1e3)")

    L_samples = []
    num_samples = min(50, len(z_traj_list))
    sample_indices = np.linspace(0, len(z_traj_list) - 1, num_samples, dtype=int)

    with torch.no_grad():
        for idx in sample_indices:
            z_sample = torch.tensor(z_traj_list[int(idx)][0], dtype=torch.float32)
            L_sample = learner.forward_L_tensor(z_sample.unsqueeze(0))[0].numpy()
            L_samples.append(L_sample)

    L_samples = np.array(L_samples)
    sample_states = np.asarray([traj[0] for traj in z_traj_list[: min(4, len(z_traj_list))]], dtype=np.float32)

    return analyze_general_model_data(
        z_learned=z_pred_array,
        z_truth=z_truth_array,
        L_matrices=L_samples,
        dimension=dim,
        system_name="RigidBody",
        analyzer=analyzer,
        verbose=verbose,
        learner=learner,
        state_samples=sample_states,
    )
