from torch.utils.data import Dataset
import numpy as np
import torch
import json
from pathlib import Path
from typing import Optional

class TrajectoryDataset(Dataset):
    """
    TRAJECTORY DATASET - Universal format supporting both legacy and standard formats.
    
    Can be initialized in two ways:
    1. Legacy: TrajectoryDataset(dataframe, model="RB", ...)
    2. Standard: TrajectoryDataset.from_standard_json(json_path, ...)
    """

    def __init__(self, dataframe=None, model = "RB", device=None, no_data_to_gpu=True, dim=10):
        # Handle case where dataframe is None (for subclass initialization)
        if dataframe is None:
            return
            
        if model == "RB":
            features = np.vstack((dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"])).transpose()
            targets  = np.vstack((dataframe["mx"], dataframe["my"], dataframe["mz"])).transpose()
        elif model == "HT":
            features = np.vstack((dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"], dataframe["old_rx"], dataframe["old_ry"], dataframe["old_rz"])).transpose()
            targets  = np.vstack((dataframe["mx"], dataframe["my"], dataframe["mz"], dataframe["rx"], dataframe["ry"], dataframe["rz"])).transpose()
        elif model in ["P3D", "K3D"]:
            features = np.vstack((dataframe["old_rx"], dataframe["old_ry"], dataframe["old_rz"], dataframe["old_mx"], dataframe["old_my"], dataframe["old_mz"])).transpose()
            targets  = np.vstack((dataframe["rx"], dataframe["ry"], dataframe["rz"], dataframe["mx"], dataframe["my"], dataframe["mz"])).transpose()
        elif model == "P2D":
            features = np.vstack((dataframe["old_rx"], dataframe["old_ry"], dataframe["old_mx"], dataframe["old_my"])).transpose()
            targets  = np.vstack((dataframe["rx"], dataframe["ry"], dataframe["mx"], dataframe["my"])).transpose()
        elif model == "Sh":
            features = np.vstack((dataframe["old_u"], dataframe["old_x"], dataframe["old_y"], dataframe["old_z"])).transpose()
            targets  = np.vstack((dataframe["u"], dataframe["x"], dataframe["y"], dataframe["z"])).transpose()
        elif model == "D":
            old_r_cols = [f"old_r{i}" for i in range(dim)]
            old_p_cols = [f"old_p{i}" for i in range(dim)]
            r_cols     = [f"r{i}" for i in range(dim)]
            p_cols     = [f"p{i}" for i in range(dim)]

            features = dataframe[old_r_cols + old_p_cols].to_numpy()
            targets  = dataframe[r_cols + p_cols].to_numpy()
        else:
            raise Exception("Unknown model.")

        mid = 0.5 * (features + targets)
        
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.mid = torch.from_numpy(mid)
        
        if no_data_to_gpu and device is not None and device.type == 'cuda':
            self.features = self.features.to(device)
            self.targets = self.targets.to(device)
            self.mid = self.mid.to(device)
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx], self.mid[idx])
    
    @classmethod
    def from_standard_json(cls, json_path: str, device=None, no_data_to_gpu=True):
        """
        Load from standard JSON format.
        
        This enables learning on arbitrary systems without hardcoding dimensions.
        
        Args:
            json_path: Path to standard format JSON file
            device: PyTorch device
            no_data_to_gpu: If False, move data to GPU
        
        Returns:
            TrajectoryDataset instance
        """
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        metadata = data["metadata"]
        trajectories = data["trajectories"]
        
        # Stack all trajectories
        z_list = [np.array(traj["z"], dtype=np.float32) for traj in trajectories]
        z_all = np.vstack(z_list)  # (total_steps, dim)
        
        # Compute z_dot via finite differences
        dt = metadata["dt"]
        z_dot_all = (z_all[1:] - z_all[:-1]) / dt
        
        # Remove last state (no corresponding z_dot)
        z_all_trimmed = z_all[:-1]
        z_all_next = z_all[1:]  # z(t+1)
        
        # Create instance
        instance = cls(dataframe=None, device=device, no_data_to_gpu=no_data_to_gpu)
        
        # Set attributes
        instance.features = torch.from_numpy(z_all_trimmed)
        instance.targets = torch.from_numpy(z_dot_all)
        
        # Compute midpoint: z_mid = 0.5*(z(t) + z(t+1))
        instance.mid = 0.5 * (z_all_trimmed + z_all_next)
        
        # Move to GPU if requested
        if not no_data_to_gpu and device is not None and device.type == 'cuda':
            instance.features = instance.features.to(device)
            instance.targets = instance.targets.to(device)
            instance.mid = instance.mid.to(device)
        
        return instance
