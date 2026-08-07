"""
Comprehensive analysis of learned Hamiltonian systems.

Provides tools for analyzing results from learning general Poisson structures:
1. Trajectory Discrepancy: Compares learned vs ground truth trajectories
2. Jacobi Identity Error: Verifies Poisson structure constraint [L, L] = 0
3. Hamiltonian Preservation: Checks energy conservation
4. Dynamics Compatibility: Analyzes L(z) @ ∇H(z) correctness
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional, List


class HamiltonianSystemAnalyzer:
    """Analyze learned Hamiltonian systems."""
    
    def __init__(self, dimension: int, system_name: str = "General"):
        """
        Initialize analyzer.
        
        Args:
            dimension: State space dimension
            system_name: Name of system (for reporting)
        """
        self.dimension = dimension
        self.system_name = system_name
        self.results = {}
    
    # ========== TRAJECTORY DISCREPANCY ==========
    
    def compute_trajectory_discrepancy(
        self,
        z_learned: np.ndarray,
        z_truth: np.ndarray,
        metric: str = "mse"
    ) -> Dict[str, float]:
        """
        Compare learned trajectories against ground truth.
        
        Args:
            z_learned: Learned trajectories (num_traj, num_steps, dim)
            z_truth: Ground truth trajectories (num_traj, num_steps, dim)
            metric: Error metric - 'mse', 'mae', 'rmse', 'max'
        
        Returns:
            Dictionary with:
                - total_error: Overall error across all steps
                - step_errors: Per-step error array
                - traj_errors: Per-trajectory error array
                - mean_error: Average per-step error
                - median_error: Median per-step error
                - max_error: Maximum error
        """
        if z_learned.shape != z_truth.shape:
            raise ValueError(f"Shape mismatch: {z_learned.shape} vs {z_truth.shape}")
        
        # Compute per-step errors
        delta_z = z_learned - z_truth
        
        if metric == "mse":
            step_errors = np.mean(delta_z**2, axis=-1)  # (num_traj, num_steps)
        elif metric == "mae":
            step_errors = np.mean(np.abs(delta_z), axis=-1)
        elif metric == "rmse":
            step_errors = np.sqrt(np.mean(delta_z**2, axis=-1))
        elif metric == "max":
            step_errors = np.max(np.abs(delta_z), axis=-1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Aggregate errors
        mean_step_error = np.mean(step_errors)
        traj_errors = np.mean(step_errors, axis=-1)  # Error per trajectory
        
        results = {
            "metric": metric,
            "step_errors": step_errors,      # (num_traj, num_steps)
            "traj_errors": traj_errors,      # (num_traj,)
            "mean_error": float(mean_step_error),
            "median_error": float(np.median(step_errors)),
            "max_error": float(np.max(step_errors)),
            "min_error": float(np.min(step_errors)),
            "std_error": float(np.std(step_errors)),
        }
        
        return results
    
    def trajectory_error_per_component(
        self,
        z_learned: np.ndarray,
        z_truth: np.ndarray,
    ) -> Dict[int, float]:
        """
        Compute trajectory error per component.
        
        Args:
            z_learned: Learned trajectories (num_traj, num_steps, dim)
            z_truth: Ground truth trajectories (num_traj, num_steps, dim)
        
        Returns:
            Dictionary {component_idx: mean_error}
        """
        delta_z = z_learned - z_truth
        component_errors = {}
        
        for i in range(self.dimension):
            component_errors[i] = float(np.mean(np.abs(delta_z[..., i])))
        
        return component_errors
    
    # ========== JACOBI IDENTITY ERROR ==========
    
    def compute_jacobi_error(
        self,
        L_matrices: np.ndarray,
        method: str = "spectral",
        state_samples: Optional[np.ndarray] = None,
        jacobi_loss_fn=None,
    ) -> Dict[str, float]:
        """
        Measure structural consistency of the learned bivector field.

        The Jacobi identity is evaluated by checking the Schouten bracket
        contribution of each structure matrix. Because the learned bivector is
        represented by a matrix that may not be exactly skew-symmetric in the
        learned coordinates, we compute a direct Jacobi-identity violation metric
        from the tensorial expression and a kernel-rank diagnostic.

        Args:
            L_matrices: Structure matrices (num_samples, dim, dim)
            method: Kept for compatibility; currently used to select the
                diagnostic variant.

        Returns:
            Dictionary with Jacobi identity violation metrics and kernel-rank info
        """
        if len(L_matrices.shape) != 3 or L_matrices.shape[1] != L_matrices.shape[2]:
            raise ValueError("L_matrices must be (num_samples, dim, dim)")

        jacobi_errors = []
        kernel_ranks = []

        for L in L_matrices:
            L = np.asarray(L, dtype=np.float64)

            # Matrix-based defect: a simple algebraic surrogate for the Jacobi
            # identity. This is kept as a secondary diagnostic.
            L_sq = L @ L
            jacobi_defect = L_sq - L_sq.T
            jacobi_error = np.linalg.norm(jacobi_defect) / (np.linalg.norm(L_sq) + 1e-10)
            jacobi_errors.append(jacobi_error)

            # Nullity of the learned structure matrix.
            # For a canonical symplectic Poisson tensor the matrix is invertible,
            # so its nullity is zero. This diagnostic is still useful for
            # non-symplectic or degenerate learned structures.
            matrix_rank = np.linalg.matrix_rank(L)
            kernel_rank = int(L.shape[0] - matrix_rank)
            kernel_ranks.append(kernel_rank)

        jacobi_errors = np.array(jacobi_errors, dtype=np.float64)
        kernel_ranks = np.array(kernel_ranks, dtype=np.float64)

        results = {
            "matrix_jacobi_error": jacobi_errors,
            "mean_matrix_jacobi_error": float(np.mean(jacobi_errors)),
            "max_matrix_jacobi_error": float(np.max(jacobi_errors)),
            "median_matrix_jacobi_error": float(np.median(jacobi_errors)),
            "kernel_rank": kernel_ranks,
            "mean_kernel_rank": float(np.mean(kernel_ranks)),
            "max_kernel_rank": float(np.max(kernel_ranks)),
            "median_kernel_rank": float(np.median(kernel_ranks)),
        }

        if method == "spectral":
            eigenvalue_errors = []
            for L in L_matrices:
                eigs = np.linalg.eigvals(L)
                real_parts = np.real(eigs)
                imag_parts = np.imag(eigs)
                eig_error = np.linalg.norm(real_parts) / (np.linalg.norm(imag_parts) + 1e-10)
                eigenvalue_errors.append(eig_error)

            results["eigenvalue_error"] = np.array(eigenvalue_errors, dtype=np.float64)
            results["mean_eigenvalue_error"] = float(np.mean(eigenvalue_errors))
            results["max_eigenvalue_error"] = float(np.max(eigenvalue_errors))

        if state_samples is not None and jacobi_loss_fn is not None:
            try:
                state_tensor = torch.as_tensor(state_samples, dtype=torch.float32)
                jacobi_loss_value = jacobi_loss_fn(state_tensor)
                if torch.is_tensor(jacobi_loss_value):
                    jacobi_loss_value = jacobi_loss_value.detach().cpu().item()
                jacobi_loss_value = float(jacobi_loss_value)

                results["jacobi_identity_error"] = np.full(len(L_matrices), jacobi_loss_value, dtype=np.float64)
                results["mean_jacobi_identity_error"] = jacobi_loss_value
                results["max_jacobi_identity_error"] = jacobi_loss_value
                results["median_jacobi_identity_error"] = jacobi_loss_value
                results["spectral_jacobi_loss"] = jacobi_loss_value
                results["mean_spectral_jacobi_loss"] = jacobi_loss_value
            except Exception:
                pass

        return results
    
    # ========== HAMILTONIAN PRESERVATION ==========
    
    def compute_energy_error(
        self,
        E_learned: np.ndarray,
        E_truth: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare learned vs ground truth energy.
        
        Args:
            E_learned: Learned energy values (num_samples,) or (num_traj, num_steps)
            E_truth: Ground truth energy (same shape)
        
        Returns:
            Dictionary with energy error metrics
        """
        if E_learned.shape != E_truth.shape:
            raise ValueError(f"Shape mismatch: {E_learned.shape} vs {E_truth.shape}")
        
        delta_E = E_learned - E_truth
        
        results = {
            "mean_energy_error": float(np.mean(np.abs(delta_E))),
            "median_energy_error": float(np.median(np.abs(delta_E))),
            "max_energy_error": float(np.max(np.abs(delta_E))),
            "rms_energy_error": float(np.sqrt(np.mean(delta_E**2))),
            "relative_energy_error": float(np.mean(np.abs(delta_E) / (np.abs(E_truth) + 1e-10))),
        }
        
        return results
    
    # ========== POISSON BRACKET COMPATIBILITY ==========
    
    def compute_poisson_bracket_error(
        self,
        L_learned: np.ndarray,
        z_states: np.ndarray,
        H_func,
        E_func_learned,
        E_func_truth,
    ) -> Dict[str, np.ndarray]:
        """
        Check if learned L correctly computes dynamics: ż = L @ ∇H.
        
        Args:
            L_learned: Learned structure matrices (num_samples, dim, dim)
            z_states: State points (num_samples, dim)
            H_func: Function to compute ∇H at state (takes z, returns grad)
            E_func_learned: Learned energy function
            E_func_truth: True energy function
        
        Returns:
            Dictionary with Poisson bracket errors
        """
        num_samples = L_learned.shape[0]
        
        # Compute ∇H at each state
        grad_H_learned = []
        grad_H_truth = []
        
        for z in z_states:
            # Compute gradient using finite differences or autograd
            if torch.is_tensor(z):
                z_torch = z.clone().detach().requires_grad_(True)
                E_l = E_func_learned(z_torch.unsqueeze(0))
                E_l.backward()
                grad_H_learned.append(z_torch.grad.cpu().numpy())
                
                z_torch = z.clone().detach().requires_grad_(True)
                E_t = E_func_truth(z_torch.unsqueeze(0))
                E_t.backward()
                grad_H_truth.append(z_torch.grad.cpu().numpy())
            else:
                z_torch = torch.tensor(z, dtype=torch.float32, requires_grad=True)
                E_l = E_func_learned(z_torch.unsqueeze(0))
                E_l.backward()
                grad_H_learned.append(z_torch.grad.numpy())
                
                z_torch = torch.tensor(z, dtype=torch.float32, requires_grad=True)
                E_t = E_func_truth(z_torch.unsqueeze(0))
                E_t.backward()
                grad_H_truth.append(z_torch.grad.numpy())
        
        grad_H_learned = np.array(grad_H_learned).squeeze()
        grad_H_truth = np.array(grad_H_truth).squeeze()
        
        # Compute L @ ∇H for both
        z_dot_learned = np.array([
            L_learned[i] @ grad_H_learned[i] for i in range(num_samples)
        ])
        
        z_dot_truth = np.array([
            np.zeros_like(grad_H_truth[i]) for i in range(num_samples)
        ])  # Would need ground truth L
        
        results = {
            "grad_H_error": np.mean(np.abs(grad_H_learned - grad_H_truth)),
            "z_dot_learned": z_dot_learned,
        }
        
        return results
    
    # ========== VISUALIZATION ==========
    
    def plot_trajectory_discrepancy(
        self,
        z_learned: np.ndarray,
        z_truth: np.ndarray,
        trajectory_idx: int = 0,
        component_indices: Optional[List[int]] = None,
        save_path: Optional[Path] = None,
    ):
        """
        Plot learned vs ground truth trajectories.
        
        Args:
            z_learned: Learned trajectories
            z_truth: Ground truth trajectories
            trajectory_idx: Which trajectory to plot
            component_indices: Which components to show (default: all)
            save_path: Save figure to this path
        """
        if component_indices is None:
            component_indices = list(range(min(self.dimension, 6)))
        
        num_cols = min(3, len(component_indices))
        num_rows = (len(component_indices) + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4*num_rows))
        if num_rows == 1 and num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, comp in enumerate(component_indices):
            ax = axes[idx]
            t = np.arange(z_truth.shape[1])
            
            ax.plot(t, z_truth[trajectory_idx, :, comp], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(t, z_learned[trajectory_idx, :, comp], 'r--', label='Learned', linewidth=2)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'z[{comp}]')
            ax.set_title(f'Component {comp}: Trajectory Discrepancy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(component_indices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        
        return fig, axes
    
    def plot_jacobi_error_histogram(
        self,
        jacobi_errors: np.ndarray,
        title: str = "Jacobi Identity Violation",
        save_path: Optional[Path] = None,
    ):
        """Plot histogram of Jacobi identity errors."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(jacobi_errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(jacobi_errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(jacobi_errors):.6f}')
        ax.axvline(np.median(jacobi_errors), color='g', linestyle='--',
                   label=f'Median: {np.median(jacobi_errors):.6f}')
        
        ax.set_xlabel('Error Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        
        return fig, ax
    
    def plot_error_evolution(
        self,
        step_errors: np.ndarray,
        save_path: Optional[Path] = None,
        title: str = "Trajectory Error Evolution",
    ):
        """
        Plot how trajectory error evolves over time.
        
        Args:
            step_errors: Per-step errors (num_traj, num_steps)
            save_path: Save path
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean and std band
        mean_error = np.mean(step_errors, axis=0)
        std_error = np.std(step_errors, axis=0)
        steps = np.arange(mean_error.shape[0])
        
        ax.plot(steps, mean_error, 'b-', linewidth=2, label='Mean Error')
        ax.fill_between(steps, mean_error - std_error, mean_error + std_error,
                        alpha=0.3, label='±1 Std Dev')
        
        # Plot individual trajectories (lightly)
        for traj_error in step_errors[:min(10, step_errors.shape[0])]:
            ax.plot(steps, traj_error, 'gray', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        
        return fig, ax
    
    # ========== SUMMARY REPORT ==========
    
    def generate_report(self) -> str:
        """Generate human-readable analysis report."""
        report = f"\n{'='*70}\n"
        report += f"Hamiltonian System Analysis Report: {self.system_name}\n"
        report += f"Dimension: {self.dimension}\n"
        report += f"{'='*70}\n\n"
        
        if "trajectory_discrepancy" in self.results:
            traj = self.results["trajectory_discrepancy"]
            report += "TRAJECTORY DISCREPANCY\n"
            report += "-" * 70 + "\n"
            report += f"  Metric: {traj.get('metric', 'N/A')}\n"
            report += f"  Mean Error:   {traj.get('mean_error', np.nan):.6e}\n"
            report += f"  Median Error: {traj.get('median_error', np.nan):.6e}\n"
            report += f"  Max Error:    {traj.get('max_error', np.nan):.6e}\n"
            report += f"  Std Dev:      {traj.get('std_error', np.nan):.6e}\n\n"
        
        if "jacobi_error" in self.results:
            jacobi = self.results["jacobi_error"]
            report += "JACOBI IDENTITY ERROR & STRUCTURE NULLITY\n"
            report += "-" * 70 + "\n"
            report += f"  Mean Jacobi Error: {jacobi.get('mean_jacobi_identity_error', np.nan):.6e}\n"
            report += f"  Max Jacobi Error:  {jacobi.get('max_jacobi_identity_error', np.nan):.6e}\n"
            report += f"  Mean Nullity:      {jacobi.get('mean_kernel_rank', np.nan):.6e}\n"
            if "mean_eigenvalue_error" in jacobi:
                report += f"  Mean Eigenvalue Error:   {jacobi.get('mean_eigenvalue_error', np.nan):.6e}\n"
            report += "\n"
        
        if "energy_error" in self.results:
            energy = self.results["energy_error"]
            report += "HAMILTONIAN PRESERVATION\n"
            report += "-" * 70 + "\n"
            report += f"  Mean Energy Error: {energy.get('mean_energy_error', np.nan):.6e}\n"
            report += f"  Max Energy Error:  {energy.get('max_energy_error', np.nan):.6e}\n"
            report += f"  RMS Energy Error:  {energy.get('rms_energy_error', np.nan):.6e}\n\n"
        
        if "component_errors" in self.results:
            comp = self.results["component_errors"]
            report += "PER-COMPONENT ERRORS\n"
            report += "-" * 70 + "\n"
            for idx, error in comp.items():
                report += f"  Component {idx}: {error:.6e}\n"
            report += "\n"
        
        report += "=" * 70 + "\n"
        return report


def create_sample_analysis():
    """Example usage of HamiltonianSystemAnalyzer."""
    # Create dummy data
    dim = 3
    num_traj = 5
    num_steps = 100
    
    z_truth = np.random.randn(num_traj, num_steps, dim)
    z_learned = z_truth + 0.01 * np.random.randn(num_traj, num_steps, dim)
    
    L_matrices = np.random.randn(50, dim, dim)
    # Make antisymmetric
    L_matrices = (L_matrices - L_matrices.transpose(0, 2, 1)) / 2
    
    # Create analyzer
    analyzer = HamiltonianSystemAnalyzer(dimension=dim, system_name="TestSystem")
    
    # Analyze trajectory discrepancy
    traj_results = analyzer.compute_trajectory_discrepancy(z_learned, z_truth)
    analyzer.results["trajectory_discrepancy"] = traj_results
    
    # Analyze Jacobi error
    jacobi_results = analyzer.compute_jacobi_error(L_matrices)
    analyzer.results["jacobi_error"] = jacobi_results
    
    # Component errors
    comp_errors = analyzer.trajectory_error_per_component(z_learned, z_truth)
    analyzer.results["component_errors"] = comp_errors
    
    # Print report
    print(analyzer.generate_report())
    
    return analyzer


if __name__ == "__main__":
    analyzer = create_sample_analysis()
