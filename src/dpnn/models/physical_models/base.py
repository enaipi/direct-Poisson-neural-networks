#This file contains the generic GeneralSystem class and model loading utilities

from scipy.optimize import fsolve
from math import *
import numpy as np
from typing import Callable, Optional, Tuple

import torch

from dpnn.models.energy_nn import EnergyNet
from dpnn.models.tensor_nn import TensorNet, JacVectorNet
from dpnn.training import DEFAULT_folder_name


def load_models(name = DEFAULT_folder_name, method = "without", mx = torch.zeros((1,1)), device="cpu"):
    """Load neural network models for energy and L/J structures.
    
    :param name: Folder name where saved models are located
    :param method: Method type - "soft", "without", or "implicit"
    :param mx: Template tensor for device/dtype inference
    :param device: Device to load models onto
    :return: Tuple of (energy_net, L_net, J_net, A)
    """
    A, J_net = None, None
    if method == "soft":
        energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)
        energy_net.eval()   
        L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)
        L_net.eval()
    elif method == "without":
        energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)
        energy_net.eval()

        obj = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)
        if isinstance(obj, torch.nn.Module): # old format
            L_net = obj
            L_net.eval()

        elif isinstance(obj, dict):
            L_type = obj.get('L_type', 'module')
            if L_type == 'constant':
                A = obj['A'].to(device)
                def L_net(z):
                    L = A - A.t()
                    return L.unsqueeze(0).repeat(z.size(0), 1, 1)
            elif L_type == 'module':
                L_net = obj['L_tensor']
                if isinstance(L_net, torch.nn.Module):
                    L_net.to(device)
                    L_net.eval()
            else:
                raise ValueError(f"Unknown L_type: {L_type}")
    elif method == "implicit":
        energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)
        energy_net.eval()
        J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)
        J_net.eval()
        J_net = J_net.to(device)
        def L_net(z):
            zeros = torch.zeros_like(mx)
            L = torch.stack([
                torch.stack([zeros, z[:, 2], -z[:, 1]], dim=1),
                torch.stack([-z[:, 2], zeros, z[:, 0]], dim=1),
                torch.stack([z[:, 1], -z[:, 0], zeros], dim=1)
            ], dim=1)
            return -L
    else:
        raise Exception("Unkonown method: ", method)
    
    return energy_net.to(device), L_net, J_net, A


class GeneralSystem(object):
    """
    Generic Poisson structure integrator for arbitrary dynamical systems.
    
    Governs evolution via: z_dot = L(z) @ grad_E(z)
    where z is an arbitrary state vector.
    
    This class decouples the mathematical structure (integration schemes) from physics
    specifics (energy and Poisson matrix), enabling code reuse across all system types.
    """
    
    def __init__(self, 
                 z_init: torch.Tensor,
                 energy_fn: Callable,
                 poisson_fn: Callable,
                 grad_energy_fn: Callable,
                 dt: float,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 verbose: bool = False):
        """
        Initialize a general dynamical system.
        
        :param z_init: Initial state vector, shape (batch_size, dim) or (dim,)
        :param energy_fn: Callable E(z) -> tensor(batch_size) or tensor(). 
                         If z has shape (batch, dim), returns (batch,). 
                         If z has shape (dim,), returns scalar.
        :param poisson_fn: Callable L(z) -> tensor(batch_size, dim, dim) or tensor(dim, dim).
                          Skew-symmetric Poisson bivector.
        :param grad_energy_fn: Callable grad_E(z) -> tensor same shape as z.
                              Energy gradient.
        :param dt: Time step for integration
        :param device: Device ("cpu" or "cuda")
        :param dtype: Data type (torch.float32 or torch.float64)
        :param verbose: Print debug info
        """
        # Ensure z_init has batch dimension
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)
        
        self.z = z_init.clone().to(device=device, dtype=dtype)
        self.z0 = z_init.clone().to(device=device, dtype=dtype)
        
        self.energy_fn = energy_fn
        self.poisson_fn = poisson_fn
        self.grad_energy_fn = grad_energy_fn
        
        self.dt = dt
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        
        self.dim = self.z.shape[-1]  # Dimension of state space
        self.batch_size = self.z.shape[0]  # Number of trajectories
        
        if self.verbose:
            print(f"GeneralSystem initialized: batch_size={self.batch_size}, dim={self.dim}")
    
    def get_E(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy at state z."""
        return self.energy_fn(z)
    
    def get_L(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Poisson bivector at state z. Returns (batch, dim, dim) tensor."""
        L = self.poisson_fn(z)
        # Ensure correct shape for batch operations
        if L.dim() == 2:  # Single (dim, dim) matrix
            L = L.unsqueeze(0).expand(z.shape[0], -1, -1)
        return L
    
    def get_grad_E(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy gradient at state z. Returns same shape as z."""
        return self.grad_energy_fn(z)
    
    def get_zdot(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute z_dot = L(z) @ grad_E(z).
        
        :param z: State vector, shape (batch_size, dim)
        :return: Time derivative, shape (batch_size, dim)
        """
        grad_E = self.get_grad_E(z)  # (batch_size, dim)
        L = self.get_L(z)             # (batch_size, dim, dim)
        
        # Matrix-vector multiplication for each batch element
        z_dot = torch.einsum('bij,bj->bi', L, grad_E)
        return z_dot
    
    def m_new_explicit_euler(self) -> torch.Tensor:
        """
        One step of explicit Euler (forward Euler).
        Simple but not structure-preserving.
        
        z_new = z_old + dt * z_dot(z_old)
        """
        z_dot = self.get_zdot(self.z)
        self.z = self.z + self.dt * z_dot
        return self.z
    
    def m_new_crank_nicolson(self, 
                             solver_iterations: int = 200,
                             tol: float = 1e-6) -> torch.Tensor:
        """
        One step of Crank-Nicolson (semi-implicit, symplectic).
        
        z_new = z_old + (dt/2) * (z_dot(z_old) + z_dot(z_new))
        Solved via fixed-point iteration.
        
        :param solver_iterations: Max iterations for implicit solve
        :param tol: Convergence tolerance (relative)
        :return: New state z_new
        """
        z_old = self.z.clone()
        z_new = z_old.clone()
        
        z_dot_old = self.get_zdot(z_old)
        
        for iteration in range(solver_iterations):
            z_prev = z_new.clone()
            z_dot_new = self.get_zdot(z_new)
            
            z_new = z_old + 0.5 * self.dt * (z_dot_old + z_dot_new)
            
            # Check convergence
            diff = torch.norm(z_new - z_prev, dim=1)
            denom = torch.norm(z_prev, dim=1) + 1e-12
            rel_error = diff / denom
            
            if torch.all(rel_error < tol):
                break
        
        else:
            # If loop completes without breaking, check which trajectories didn't converge
            not_converged = (rel_error >= tol)
            if not_converged.any():
                if self.verbose:
                    print(f"CN: {not_converged.sum().item()} trajectories did not converge (max iterations).")
        
        self.z = z_new
        return z_new
    
    def m_new_implicit_midpoint(self, 
                                solver_iterations: int = 200,
                                tol: float = 1e-6) -> torch.Tensor:
        """
        One step of Implicit Midpoint Rule (IMR, fully symplectic).
        
        z_new = z_old + dt * z_dot((z_old + z_new) / 2)
        Solved via fixed-point iteration.
        
        :param solver_iterations: Max iterations for implicit solve
        :param tol: Convergence tolerance (relative)
        :return: New state z_new
        """
        z_old = self.z.clone()
        z_new = z_old.clone()
        
        for iteration in range(solver_iterations):
            z_prev = z_new.clone()
            z_mid = 0.5 * (z_old + z_new)
            z_dot_mid = self.get_zdot(z_mid)
            
            z_new = z_old + self.dt * z_dot_mid
            
            # Check convergence
            diff = torch.norm(z_new - z_prev, dim=1)
            denom = torch.norm(z_prev, dim=1) + 1e-12
            rel_error = diff / denom
            
            if torch.all(rel_error < tol):
                break
        
        else:
            not_converged = (rel_error >= tol)
            if not_converged.any():
                if self.verbose:
                    print(f"IMR: {not_converged.sum().item()} trajectories did not converge (max iterations).")
        
        self.z = z_new
        return z_new
    
    def m_new_rk4(self) -> torch.Tensor:
        """
        One step of 4th-order Runge-Kutta (explicit, not structure-preserving but accurate).
        
        :return: New state z_new
        """
        z_old = self.z.clone()
        
        k1 = self.get_zdot(z_old)
        k2 = self.get_zdot(z_old + 0.5 * self.dt * k1)
        k3 = self.get_zdot(z_old + 0.5 * self.dt * k2)
        k4 = self.get_zdot(z_old + self.dt * k3)
        
        z_new = z_old + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        self.z = z_new
        return z_new
    
    def m_new(self, method: str = "crank_nicolson", **kwargs) -> torch.Tensor:
        """
        Generic integrator step dispatcher.
        
        :param method: Integration method ("euler", "crank_nicolson", "implicit_midpoint", "rk4")
        :param kwargs: Method-specific arguments
        :return: New state
        """
        if method == "euler":
            return self.m_new_explicit_euler()
        elif method == "crank_nicolson" or method == "cn":
            return self.m_new_crank_nicolson(**kwargs)
        elif method == "implicit_midpoint" or method == "imr":
            return self.m_new_implicit_midpoint(**kwargs)
        elif method == "rk4":
            return self.m_new_rk4()
        else:
            raise ValueError(f"Unknown integration method: {method}")

