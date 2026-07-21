"""Error computation and compatibility analysis functions."""

import numpy as np
import pandas as pd
from .data_utils import load_normalized_Ls


def compat_error3D(L, mx, my, mz):
    """
    Compute 3D compatibility error between L and Jacobian.
    
    For RigidBody, computes ||[L, J]||_F where J is the Jacobian matrix
    and [L, J] = LJ - JL is the commutator.
    """
    if len(L) != len(mx):
        raise Exception("L and mx must have same length")
    
    errors = []
    for i in range(len(mx)):
        J = np.array([
            [0.0, -mz[i], my[i]],
            [mz[i], 0.0, -mx[i]],
            [-my[i], mx[i], 0.0]
        ])
        
        commutator = np.matmul(L[i], J) - np.matmul(J, L[i])
        error = np.linalg.norm(commutator)
        errors.append(error)
    
    return np.array(errors)


def compat_error_superintegrable(L, mxs, mys, mzs, rxs, rys, rzs):
    """
    Compute compatibility error for heavy top.
    
    Checks that L matrix satisfies specific structural constraints.
    """
    errors = []
    
    for i in range(len(mxs)):
        J = np.array([
            [0.0, -mzs[i], mys[i], 0.0, -rzs[i], rys[i]],
            [mzs[i], 0.0, -mxs[i], rzs[i], 0.0, -rxs[i]],
            [-mys[i], mxs[i], 0.0, -rys[i], rxs[i], 0.0],
            [0.0, -rzs[i], rys[i], 0.0, 0.0, 0.0],
            [rzs[i], 0.0, -rxs[i], 0.0, 0.0, 0.0],
            [-rys[i], rxs[i], 0.0, 0.0, 0.0, 0.0]
        ])
        
        commutator = np.matmul(L[i], J) - np.matmul(J, L[i])
        error = np.linalg.norm(commutator)
        errors.append(error)
    
    return np.array(errors)


def compat_error_6d_separable(L, qs, ps):
    """Compute compatibility error for 6D separable systems (particle 3D)."""
    errors = []
    
    J = np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    ])
    
    for i in range(len(L)):
        commutator = np.matmul(L[i], J) - np.matmul(J, L[i])
        error = np.linalg.norm(commutator)
        errors.append(error)
    
    return np.array(errors)


def compute_spectrum_errors(Ls, model_type):
    """
    Compute spectrum error based on model type.
    
    For RigidBody: eigenvalues should be purely imaginary (Hamiltonian structure)
    """
    if model_type == "RB":
        σ_errors = []
        for i in range(len(Ls)):
            eigenvalues = np.linalg.eig(Ls[i])[0]
            # For Poisson structure, eigenvalues should satisfy specific properties
            σ = np.sort_complex(1.j * eigenvalues)
            # Compute some norm of eigenvalues
            error = np.linalg.norm(σ)
            σ_errors.append(error)
        return np.array(σ_errors)
    else:
        return None
