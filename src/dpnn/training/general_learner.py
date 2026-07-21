"""
GeneralSystemLearner: Universal learning pipeline for arbitrary dynamical systems.

This learner works with any system described by SystemSpec, eliminating the need
to hardcode system-specific logic in the training pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import einsum
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import time

from dpnn.models.energy_nn import EnergyNet
from dpnn.models.tensor_nn import TensorNet, JacVectorNet
from dpnn.system_spec import SystemSpec, get_system_spec
from dpnn.data.standard_format import StandardTrajectoryDataset


class GeneralSystemLearner:
    """
    Universal learner for arbitrary dynamical systems.
    
    Learns:
    1. Energy network E(z) - outputs energy scalar
    2. Optionally: Poisson structure tensor L(z) - outputs antisymmetric matrix
    3. System follows: z_dot = L(z) @ grad_E(z)
    
    Works for any system without hardcoding dimensions or model types.
    """
    
    def __init__(self,
                 system_spec: SystemSpec,
                 batch_size: int = 32,
                 neurons: int = 64,
                 layers: int = 2,
                 device: str = 'cpu',
                 dropout_rate: float = 0.0,
                 quad_features: bool = False,
                 jacobi_loss_mode: str = "exact",
                 hutchinson_samples: int = 3):
        """
        Initialize learner for a system.
        
        Args:
            system_spec: SystemSpec describing the system
            batch_size: Training batch size
            neurons: Neurons per layer in networks
            layers: Number of layers
            device: PyTorch device
            dropout_rate: Dropout rate in networks
            quad_features: Add quadratic features to energy net
            jacobi_loss_mode: "exact" or "soft" or "implicit"
            hutchinson_samples: Samples for stochastic Jacobian estimation
        """
        self.system_spec = system_spec
        self.dim = system_spec.dimension
        self.device = device
        self.batch_size = batch_size
        self.jacobi_loss_mode = jacobi_loss_mode
        self.hutchinson_samples = hutchinson_samples
        
        print(f"Initializing GeneralSystemLearner for {system_spec.name} (dim={self.dim})")
        
        # Energy network - always needed
        self.energy_net = EnergyNet(
            self.dim, neurons, layers, batch_size,
            dropout_rate=dropout_rate,
            quad_features=quad_features
        ).to(device)
        
        # Jacobian-vector network (for learning Poisson structure if needed)
        self.jac_vec_net = JacVectorNet(
            self.dim, neurons, layers, batch_size,
            dropout_rate=dropout_rate
        ).to(device)
        
        # Entropy network (optional, for dissipative systems)
        self.entropy_net = EnergyNet(
            self.dim, neurons, layers, batch_size,
            dropout_rate=dropout_rate,
            quad_features=quad_features
        ).to(device)
        
        # Tensor network for learning L(z) if structure_tensor == "learned"
        if system_spec.structure_tensor == "learned":
            self.tensor_net = TensorNet(
                self.dim, neurons, layers, batch_size,
                dropout_rate=dropout_rate
            ).to(device)
        else:
            self.tensor_net = None
    
    def get_poisson_structure(self, z: torch.Tensor) -> torch.Tensor:
        """
        Get Poisson structure tensor L(z).
        
        Args:
            z: State tensor, shape (batch_size, dim)
        
        Returns:
            L: Shape (batch_size, dim, dim), antisymmetric matrices
        """
        if self.system_spec.structure_tensor == "learned":
            return self.tensor_net(z)
        
        elif self.system_spec.poisson_bracket_type == "canonical":
            # Canonical symplectic structure: [[0, I], [-I, 0]]
            batch_size = z.shape[0]
            half_dim = self.dim // 2
            
            Z = torch.zeros(batch_size, self.dim, self.dim, device=z.device, dtype=z.dtype)
            Z[:, :half_dim, half_dim:] = torch.eye(half_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
            Z[:, half_dim:, :half_dim] = -torch.eye(half_dim, device=z.device, dtype=z.dtype).unsqueeze(0)
            return Z
        
        elif self.system_spec.poisson_bracket_type == "rigid_body":
            # Rigid body: -L for cross-product, L[i,j,k] = -eps_{ijk} z_k
            batch_size = z.shape[0]
            Z = torch.zeros(batch_size, self.dim, self.dim, device=z.device, dtype=z.dtype)
            
            if self.dim == 3:
                # 3D angular momentum: skew-symmetric from cross product
                Z[:, 0, 1] = -z[:, 2]
                Z[:, 0, 2] = z[:, 1]
                Z[:, 1, 0] = z[:, 2]
                Z[:, 1, 2] = -z[:, 0]
                Z[:, 2, 0] = -z[:, 1]
                Z[:, 2, 1] = z[:, 0]
                return -Z
            else:
                raise NotImplementedError(f"Rigid body structure for dim={self.dim}")
        
        else:
            raise ValueError(f"Unknown poisson_bracket_type: {self.system_spec.poisson_bracket_type}")
    
    def compute_energy_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute energy gradient via autograd.
        
        Args:
            z: State tensor, shape (batch_size, dim), requires_grad=True
        
        Returns:
            grad_E: Shape (batch_size, dim), dE/dz
        """
        z_in = z.clone().requires_grad_(True)
        E = self.energy_net(z_in)
        E_sum = E.sum()
        E_sum.backward()
        return z_in.grad
    
    def compute_z_dot_pred(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict z_dot = L(z) @ grad_E(z).
        
        Args:
            z: State tensor, shape (batch_size, dim)
        
        Returns:
            z_dot_pred: Shape (batch_size, dim)
        """
        z_req = z.clone().requires_grad_(True)
        grad_E = self.compute_energy_gradient(z_req)
        L = self.get_poisson_structure(z)
        
        # z_dot = L @ grad_E, einsum is efficient for batch
        z_dot_pred = einsum('bij,bj->bi', L, grad_E)
        return z_dot_pred
    
    def loss_dynamics(self, z: torch.Tensor, z_dot_target: torch.Tensor) -> torch.Tensor:
        """
        Dynamics loss: MSE between predicted and target z_dot.
        
        Args:
            z: State, shape (batch_size, dim)
            z_dot_target: Target velocity, shape (batch_size, dim)
        
        Returns:
            Loss scalar
        """
        z_dot_pred = self.compute_z_dot_pred(z)
        return torch.nn.functional.mse_loss(z_dot_pred, z_dot_target)
    
    def loss_jacobi(self, z: torch.Tensor) -> torch.Tensor:
        """
        Jacobi identity loss (soft constraint on Poisson structure).
        
        For canonical structure, Jacobi is automatically satisfied.
        For learned structure, we add a soft constraint.
        
        Returns:
            Loss scalar (0 if structure is predefined)
        """
        if self.system_spec.structure_tensor != "learned":
            return torch.tensor(0.0, device=self.device)
        
        # For learned structure, compute Jacobi identity constraint
        # [L, L] = 0 constraint (soft)
        # This is complex, so for now return 0
        # TODO: Implement Jacobi identity constraint
        return torch.tensor(0.0, device=self.device)
    
    def train_epoch(self,
                   data_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   jacobi_weight: float = 0.0) -> float:
        """
        Train for one epoch.
        
        Args:
            data_loader: DataLoader with (z, z_dot) batches
            optimizer: PyTorch optimizer
            jacobi_weight: Weight for Jacobi loss
        
        Returns:
            Average loss for epoch
        """
        self.energy_net.train()
        if self.tensor_net:
            self.tensor_net.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            z, z_dot_target, _ = batch  # z_mid not used here
            z = z.to(self.device)
            z_dot_target = z_dot_target.to(self.device)
            
            optimizer.zero_grad()
            
            # Main loss: dynamics
            loss_dyn = self.loss_dynamics(z, z_dot_target)
            
            # Optional: Jacobi loss
            loss_jac = self.loss_jacobi(z)
            
            # Total loss
            loss = loss_dyn + jacobi_weight * loss_jac
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self,
             data_path: str,
             epochs: int = 10,
             learning_rate: float = 1e-4,
             jacobi_weight: float = 0.0,
             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            data_path: Path to standard format JSON dataset
            epochs: Number of training epochs
            learning_rate: Adam learning rate
            jacobi_weight: Weight for Jacobi constraint loss
            save_path: Optional path to save trained model
        
        Returns:
            Training history dict
        """
        # Load data
        print(f"Loading data from {data_path}")
        dataset = StandardTrajectoryDataset(data_path, system_spec=self.system_spec, device=self.device)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        params = list(self.energy_net.parameters())
        if self.tensor_net:
            params += list(self.tensor_net.parameters())
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # Training loop
        history = {
            "epochs": [],
            "losses": [],
        }
        
        print(f"Training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            loss = self.train_epoch(data_loader, optimizer, jacobi_weight=jacobi_weight)
            history["epochs"].append(epoch)
            history["losses"].append(loss)
            
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.1f}s")
        
        # Save if requested
        if save_path:
            self.save(save_path)
            print(f"Model saved to {save_path}")
        
        return history
    
    def save(self, path: str):
        """Save trained model."""
        checkpoint = {
            "system_spec": self.system_spec.to_dict(),
            "energy_net": self.energy_net.state_dict(),
            "entropy_net": self.entropy_net.state_dict(),
            "jac_vec_net": self.jac_vec_net.state_dict(),
        }
        if self.tensor_net:
            checkpoint["tensor_net"] = self.tensor_net.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.energy_net.load_state_dict(checkpoint["energy_net"])
        self.entropy_net.load_state_dict(checkpoint["entropy_net"])
        self.jac_vec_net.load_state_dict(checkpoint["jac_vec_net"])
        
        if self.tensor_net and "tensor_net" in checkpoint:
            self.tensor_net.load_state_dict(checkpoint["tensor_net"])
    
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict z_dot for given states.
        
        Args:
            z: States, shape (batch_size, dim)
        
        Returns:
            z_dot_pred: Shape (batch_size, dim)
        """
        self.energy_net.eval()
        if self.tensor_net:
            self.tensor_net.eval()
        
        with torch.no_grad():
            z_dot = self.compute_z_dot_pred(z.to(self.device))
        
        return z_dot
