"""Unified postprocessing pipeline for all examples."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

from dpnn.comparison import plot_training_errors
from .hamiltonian_analysis import HamiltonianSystemAnalyzer
from .general_analysis import analyze_general_model_data
from .error_computation import compat_error3D, compat_error_superintegrable, compat_error_6d_separable
from .data_utils import load_dataframes, load_normalized_Ls
from .model_generation import load_L_model_from_folder


def run_postprocessing_analysis(
    folder_name: str,
    model: str = "RB",
    methods: List[str] = None,
    show_plots: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """
    Run complete postprocessing analysis for learned Hamiltonian systems.
    
    Computes and displays:
    1. Training error curves (movement and Jacobi loss)
    2. Trajectory fit errors (learned vs ground truth)
    3. Jacobi identity errors (L matrix antisymmetry)
    4. Model compatibility errors (system-specific constraints)
    
    Args:
        folder_name: Folder with results (containing data/, saved_models/)
        model: Model type (RB, HT, P2D, P3D, K3D, Sh, D)
        methods: List of methods to analyze (default: ["without", "soft", "implicit"])
        show_plots: Whether to display matplotlib plots
        verbose: Whether to print the shared general-analysis report for each method
    
    Returns:
        Dict with results for each method
    """
    if methods is None:
        methods = ["without", "soft", "implicit"]
    
    # Create namespace object for plot_training_errors
    class Args:
        def __init__(self):
            self.folder_name = folder_name
            self.without = "without" in methods
            self.soft = "soft" in methods
            self.implicit = "implicit" in methods
            self.model = model
    
    args = Args()
    results = {}
    
    # ========================================================================
    # PART 1: Plot training errors from CSV files
    # ========================================================================
    print(f"\n{'='*70}")
    print("TRAINING ERROR CURVES")
    print(f"{'='*70}")
    print(f"Plotting training and validation errors from {folder_name}/data/")
    
    try:
        plot_training_errors(args)
        if not show_plots:
            import matplotlib.pyplot as plt
            plt.close('all')  # Close plots if not showing
    except Exception as e:
        print(f"  Warning: Could not plot training errors: {e}")
    
    # ========================================================================
    # PART 2: Load data and compute trajectory/Jacobi errors for each method
    # ========================================================================
    print(f"\n{'='*70}")
    print("TRAJECTORY FIT AND JACOBI ERRORS")
    print(f"{'='*70}")
    
    try:
        # Load all dataframes (includes dataset as ground truth)
        dfs = load_dataframes(folder_name, plot_steps=None)
        
        # Look for ground truth data (saved as 'dataset')
        if 'dataset' not in dfs or dfs['dataset']['df'] is None:
            print("  Warning: Ground truth data not found. Skipping trajectory/Jacobi analysis.")
            return results
        
        gt_df = dfs['dataset']['df']
        
        # Get state columns based on model
        state_cols = _get_state_columns(model)
        z_truth = gt_df[state_cols].values if state_cols else None
        
        if z_truth is None:
            print(f"  Warning: Could not find state columns for model {model}")
            return results
        
        # Initialize analyzer
        dim = len(state_cols)
        analyzer = HamiltonianSystemAnalyzer(dimension=dim, system_name=model)
        
        # Analyze each method
        for method in methods:
            method_key = _map_method_name(method)
            
            if method_key not in dfs or dfs[method_key]['df'] is None:
                print(f"\n  Method '{method}' not found. Skipping...")
                continue
            
            print(f"\n--- Method: {method} ---")
            method_results = {}
            
            learned_df = dfs[method_key]['df']
            z_learned = learned_df[state_cols].values if state_cols else None
            
            if z_learned is None:
                print(f"    Warning: Could not extract state from learned data")
                continue

            # GENERAL ANALYSIS (trajectory fit, Jacobi identity, component errors)
            try:
                Ls = load_normalized_Ls(learned_df, dim)
                if Ls is not None and len(Ls) > 0:
                    print("\n  Shared general analysis:")
                    L_func_loaded = load_L_model_from_folder(folder_name, method, dim)
                    class LearnerWrapper:
                        def __init__(self, fn):
                            self.forward_L_tensor = fn
                    learner_obj = LearnerWrapper(L_func_loaded) if L_func_loaded is not None else None

                    method_analyzer = analyze_general_model_data(
                        z_learned=z_learned,
                        z_truth=z_truth,
                        L_matrices=Ls,
                        dimension=dim,
                        system_name=model,
                        analyzer=analyzer,
                        verbose=verbose,
                        learner=learner_obj,
                    )
                    method_results["trajectory_error"] = method_analyzer.results["trajectory_discrepancy"]
                    method_results["jacobi_error"] = method_analyzer.results["jacobi_error"]
                    method_results["component_errors"] = method_analyzer.results.get("component_errors", {})
                    print(f"  Trajectory RMSE:")
                    print(f"    - Mean:   {method_results['trajectory_error']['mean_error']:.6e}")
                    print(f"    - Median: {method_results['trajectory_error']['median_error']:.6e}")
                    print(f"    - Max:    {method_results['trajectory_error']['max_error']:.6e}")
                    print(f"  Jacobi Identity Error:")
                    print(f"    - Mean Jacobi error: {method_results['jacobi_error']['mean_jacobi_identity_error']:.6e}")
                    print(f"    - Max Jacobi error:  {method_results['jacobi_error']['max_jacobi_identity_error']:.6e}")
                    print(f"    - Mean kernel rank:  {method_results['jacobi_error']['mean_kernel_rank']:.6e}")
                    if 'mean_eigenvalue_error' in method_results['jacobi_error']:
                        print(f"    - Eigenvalue purity: {method_results['jacobi_error']['mean_eigenvalue_error']:.6e}")
                else:
                    print("    Warning: No L matrices available for general analysis")
            except Exception as e:
                print(f"    Warning: Could not compute general analysis: {e}")
            
            # MODEL-SPECIFIC COMPATIBILITY ERRORS
            try:
                compat_error = _compute_compatibility_error(
                    model, learned_df, z_learned
                )
                if compat_error is not None:
                    method_results["compatibility_error"] = compat_error
                    print(f"  Compatibility Error:")
                    print(f"    - Mean:   {np.mean(compat_error):.6e}")
                    print(f"    - Max:    {np.max(compat_error):.6e}")
            except Exception as e:
                print(f"    Warning: Could not compute compatibility error: {e}")
            
            results[method] = method_results
    
    except Exception as e:
        print(f"  Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("POSTPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Analyzed {len(results)} methods: {list(results.keys())}")
    print(f"Results saved in: {folder_name}/")
    
    return results


def _get_state_columns(model: str) -> Optional[List[str]]:
    """Get state variable column names for model type."""
    state_columns = {
        "RB": ["mx", "my", "mz"],                              # Rigid body: 3D angular momentum
        "HT": ["mx", "my", "mz", "rx", "ry", "rz"],            # Heavy top: momentum + position
        "P3D": ["mx", "my", "mz", "rx", "ry", "rz"],           # 3D particle: momentum + position
        "P2D": ["mx", "my", "rx", "ry"],                       # 2D particle
        "K3D": ["mx", "my", "mz", "rx", "ry", "rz"],           # 3D Kepler
        "Sh": ["mu", "rx", "ry", "rz"],                        # Shivamoggi: scalar momentum + position
    }
    return state_columns.get(model)


def _map_method_name(method: str) -> str:
    """Map method name to dataframe key."""
    mapping = {
        "without": "without",
        "soft": "soft",
        "implicit": "implicit",
    }
    return mapping.get(method, method)


def _compute_compatibility_error(
    model: str,
    learned_df: pd.DataFrame,
    z_learned: np.ndarray
) -> Optional[np.ndarray]:
    """Compute model-specific compatibility error."""
    try:
        if model == "RB":
            # Rigid body: check [L, J] = 0 where J is Jacobian structure
            mx = z_learned[:, 0]
            my = z_learned[:, 1]
            mz = z_learned[:, 2]
            Ls = load_normalized_Ls(learned_df, 3)
            if Ls is None or len(Ls) == 0:
                return None
            return compat_error3D(Ls, mx, my, mz)
        
        elif model == "HT":
            # Heavy top: superintegrable constraint
            mx = z_learned[:, 0]
            my = z_learned[:, 1]
            mz = z_learned[:, 2]
            rx = z_learned[:, 3]
            ry = z_learned[:, 4]
            rz = z_learned[:, 5]
            Ls = load_normalized_Ls(learned_df, 6)
            if Ls is None or len(Ls) == 0:
                return None
            return compat_error_superintegrable(Ls, mx, my, mz, rx, ry, rz)
        
        elif model in ["P3D", "K3D"]:
            # Separable systems
            qs = z_learned[:, 3:6]  # positions
            ps = z_learned[:, 0:3]  # momenta
            Ls = load_normalized_Ls(learned_df, 6)
            if Ls is None or len(Ls) == 0:
                return None
            return compat_error_6d_separable(Ls, qs, ps)
        
        else:
            return None
    
    except Exception as e:
        print(f"      Warning: Could not compute compatibility error: {e}")
        return None
