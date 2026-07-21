"""Model loading and Poisson structure generation functions."""

import torch
import numpy as np
from pathlib import Path


def load_learned_models(folder_name, model_type, load_energy=True):
    """
    Load learned models from checkpoint files.
    
    Args:
        folder_name: Path to folder with saved models
        model_type: Model type (RB, HT, P2D, P3D, K3D, Sh)
        load_energy: Whether to load energy networks
    
    Returns:
        Dict with model names as keys and model info as values
    """
    models = {}
    
    checkpoint_methods = {
        'Learned Without': 'learner_without',
        'Learned Soft': 'learner_soft',
        'Learned Implicit': 'learner_implicit',
    }
    
    for name, prefix in checkpoint_methods.items():
        # Try to load old-format learner checkpoints
        checkpoint_path = Path(folder_name) / f"{prefix}.pt"
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                models[name] = {
                    'checkpoint': checkpoint,
                    'type': 'old_learner',
                }
                print(f"Loaded old learner: {name} from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Could not load {checkpoint_path}: {e}")
    
    return models


def load_general_learner(checkpoint_path):
    """
    Load a GeneralSystemLearner from checkpoint.
    
    Args:
        checkpoint_path: Path to learner checkpoint
    
    Returns:
        Loaded GeneralSystemLearner instance or None
    """
    try:
        from ..training.general_learner import GeneralSystemLearner
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        learner = GeneralSystemLearner(
            system_spec=checkpoint['system_spec'],
            hidden_layers=checkpoint.get('hidden_layers', 64),
            batch_size=checkpoint.get('batch_size', 32),
        )
        learner.load_state_dict(checkpoint['state_dict'])
        return learner
    except Exception as e:
        print(f"Warning: Could not load GeneralSystemLearner: {e}")
        return None


def get_poisson_structure(model, z, model_type='old_learner'):
    """
    Get Poisson structure L from model.
    
    Args:
        model: Loaded model (old Learner or GeneralSystemLearner)
        z: State tensor of shape (batch_size, dim)
        model_type: 'old_learner' or 'general_learner'
    
    Returns:
        L tensor of shape (batch_size, dim, dim)
    """
    if model_type == 'old_learner':
        # Old Learner has forward_L_tensor method
        if hasattr(model, 'forward_L_tensor'):
            return model.forward_L_tensor(z)
        else:
            raise ValueError("Old learner must have forward_L_tensor method")
    
    elif model_type == 'general_learner':
        # GeneralSystemLearner has get_poisson_structure method
        if hasattr(model, 'get_poisson_structure'):
            return model.get_poisson_structure(z)
        else:
            raise ValueError("GeneralSystemLearner must have get_poisson_structure method")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_E_points(args, energy_model, model_type="RB", n_points=20):
    """
    Generate a 3D mesh of energy values.
    
    Args:
        args: Arguments with folder_name
        energy_model: Trained energy network
        model_type: Model type (RB or HT)
        n_points: Number of points per dimension
    
    Returns:
        Grid tensors and energy values
    """
    if model_type == "RB":
        # RigidBody: mx, my are in [-2, 2]
        mx = np.linspace(-2, 2, n_points)
        my = np.linspace(-2, 2, n_points)
        MX, MY = np.meshgrid(mx, my)
        
        # Compute mz to satisfy constraint
        MZ = 2 - np.sqrt(4 - MX**2 - MY**2)
        
        # Convert to tensors
        MX_t = torch.tensor(MX, dtype=torch.float32)
        MY_t = torch.tensor(MY, dtype=torch.float32)
        MZ_t = torch.tensor(MZ, dtype=torch.float32)
        
        # Stack into batch
        z = torch.stack([
            MX_t.reshape(-1),
            MY_t.reshape(-1),
            MZ_t.reshape(-1)
        ], dim=1)
        
        with torch.no_grad():
            E = energy_model(z).reshape(MX.shape)
        
        return MX_t, MY_t, MZ_t, E
    
    elif model_type == "HT":
        # Heavy Top: similar structure but 6D
        mx = np.linspace(-2, 2, n_points)
        my = np.linspace(-2, 2, n_points)
        MX, MY = np.meshgrid(mx, my)
        MZ = 2 - np.sqrt(np.maximum(4 - MX**2 - MY**2, 0))
        
        MX_t = torch.tensor(MX, dtype=torch.float32)
        MY_t = torch.tensor(MY, dtype=torch.float32)
        MZ_t = torch.tensor(MZ, dtype=torch.float32)
        
        # Assume rx=ry=rz=0 for visualization
        RX_t = torch.zeros_like(MX_t)
        RY_t = torch.zeros_like(MY_t)
        RZ_t = torch.zeros_like(MZ_t)
        
        z = torch.stack([
            MX_t.reshape(-1),
            MY_t.reshape(-1),
            MZ_t.reshape(-1),
            RX_t.reshape(-1),
            RY_t.reshape(-1),
            RZ_t.reshape(-1)
        ], dim=1)
        
        with torch.no_grad():
            E = energy_model(z).reshape(MX.shape)
        
        return MX_t, MY_t, MZ_t, RX_t, RY_t, RZ_t, E
    
    else:
        raise ValueError(f"Model {model_type} not implemented for E points")


def generate_L_points(args, L_tensor_model, model_type="RB", n_points=20):
    """
    Generate a 3D mesh of Poisson structure values.
    
    Args:
        args: Arguments with folder_name
        L_tensor_model: Trained L tensor network
        model_type: Model type (RB or HT)
        n_points: Number of points per dimension
    
    Returns:
        Grid tensors and L matrix values
    """
    if model_type == "RB":
        mx = np.linspace(-2, 2, n_points)
        my = np.linspace(-2, 2, n_points)
        MX, MY = np.meshgrid(mx, my)
        MZ = 2 - np.sqrt(np.maximum(4 - MX**2 - MY**2, 0))
        
        MX_t = torch.tensor(MX, dtype=torch.float32, requires_grad=True)
        MY_t = torch.tensor(MY, dtype=torch.float32, requires_grad=True)
        MZ_t = torch.tensor(MZ, dtype=torch.float32, requires_grad=True)
        
        z = torch.stack([
            MX_t.reshape(-1),
            MY_t.reshape(-1),
            MZ_t.reshape(-1)
        ], dim=1)
        
        with torch.no_grad():
            L = L_tensor_model(z)
        
        return MX_t, MY_t, MZ_t, L
    
    elif model_type == "HT":
        mx = np.linspace(-2, 2, n_points)
        my = np.linspace(-2, 2, n_points)
        MX, MY = np.meshgrid(mx, my)
        MZ = 2 - np.sqrt(np.maximum(4 - MX**2 - MY**2, 0))
        
        MX_t = torch.tensor(MX, dtype=torch.float32, requires_grad=True)
        MY_t = torch.tensor(MY, dtype=torch.float32, requires_grad=True)
        MZ_t = torch.tensor(MZ, dtype=torch.float32, requires_grad=True)
        RX_t = torch.zeros_like(MX_t)
        RY_t = torch.zeros_like(MY_t)
        RZ_t = torch.zeros_like(MZ_t)
        
        z = torch.stack([
            MX_t.reshape(-1),
            MY_t.reshape(-1),
            MZ_t.reshape(-1),
            RX_t.reshape(-1),
            RY_t.reshape(-1),
            RZ_t.reshape(-1)
        ], dim=1)
        
        with torch.no_grad():
            L = L_tensor_model(z)
        
        return MX_t, MY_t, MZ_t, RX_t, RY_t, RZ_t, L
    
    else:
        raise ValueError(f"Model {model_type} not implemented for L points")
