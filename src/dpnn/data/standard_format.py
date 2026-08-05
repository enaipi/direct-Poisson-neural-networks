"""
Data format tools for standard trajectory data format.

Supports loading, converting, and validating trajectory data in standard format.
Separates data I/O from the learning pipeline.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class TrajectoryMetadata:
    """Metadata for a trajectory dataset."""
    system_name: str
    dimension: int
    dt: float
    num_trajectories: int
    trajectory_length: Optional[int] = None
    units: Dict[str, str] = None
    
    def __post_init__(self):
        if self.units is None:
            self.units = {}


@dataclass
class Trajectory:
    """Single trajectory data."""
    z: np.ndarray  # shape (steps, dimension)
    t: Optional[np.ndarray] = None  # time array, shape (steps,)
    id: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def compute_z_dot(self, dt: float) -> np.ndarray:
        """Compute time derivative via finite differences."""
        return (self.z[1:] - self.z[:-1]) / dt
    
    def validate(self, expected_dim: int):
        """Check trajectory has correct dimension."""
        assert self.z.ndim == 2, f"z must be 2D, got shape {self.z.shape}"
        assert self.z.shape[1] == expected_dim, \
            f"Expected dimension {expected_dim}, got {self.z.shape[1]}"


class StandardDatasetLoader:
    """Load trajectory data from standard format (JSON/HDF5/CSV)."""
    
    @staticmethod
    def load_json(path: str) -> Tuple[TrajectoryMetadata, List[Trajectory]]:
        """
        Load dataset from JSON format.
        
        Expected format:
        {
            "metadata": {
                "system_name": "RigidBody",
                "dimension": 3,
                "dt": 0.1,
                "num_trajectories": 10,
                "units": {"state": "rad/s"}
            },
            "trajectories": [
                {
                    "id": 0,
                    "z": [[...], [...], ...],
                    "t": [0, 0.1, 0.2, ...],
                    "metadata": {"Ix": 1.0, "initial_energy": 1.5}
                },
                ...
            ]
        }
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Parse metadata
        meta_dict = data["metadata"]
        metadata = TrajectoryMetadata(
            system_name=meta_dict["system_name"],
            dimension=meta_dict["dimension"],
            dt=meta_dict["dt"],
            num_trajectories=meta_dict.get("num_trajectories"),
            trajectory_length=meta_dict.get("trajectory_length"),
            units=meta_dict.get("units", {}),
        )
        
        # Parse trajectories
        trajectories = []
        for traj_dict in data["trajectories"]:
            traj = Trajectory(
                z=np.array(traj_dict["z"], dtype=np.float32),
                t=np.array(traj_dict.get("t")) if "t" in traj_dict else None,
                id=traj_dict.get("id", 0),
                metadata=traj_dict.get("metadata", {}),
            )
            traj.validate(metadata.dimension)
            trajectories.append(traj)
        
        return metadata, trajectories
    
    @staticmethod
    def load_csv_trajectories(path: str, 
                             system_spec: 'SystemSpec',
                             dt: float = 0.1) -> Tuple[TrajectoryMetadata, List[Trajectory]]:
        """
        Convert old CSV format (with old_* columns) to standard format.
        
        Args:
            path: Path to CSV file
            system_spec: SystemSpec for the system (provides dimension info)
            dt: Time step
        
        Returns:
            (metadata, trajectories)
        """
        from dpnn.system_spec import SystemSpec
        
        df = pd.read_csv(path, dtype=np.float32)
        dim = system_spec.dimension
        
        metadata = TrajectoryMetadata(
            system_name=system_spec.name,
            dimension=dim,
            dt=dt,
            num_trajectories=1,
            trajectory_length=len(df),
        )
        
        # Reconstruct z from old/new state pairs
        z_list = []
        
        # Get column names for this system
        if system_spec.name == "RigidBody":
            cols = ["mx", "my", "mz"]
        elif system_spec.name == "HeavyTop":
            cols = ["mx", "my", "mz", "rx", "ry", "rz"]
        elif system_spec.name in ["Particle3D"]:
            cols = ["rx", "ry", "rz", "mx", "my", "mz"]
        elif system_spec.name == "Particle2D":
            cols = ["rx", "ry", "mx", "my"]
        else:
            raise ValueError(f"CSV conversion not supported for {system_spec.name}")
        
        # Use 'old_*' columns to get initial state
        old_state = np.array([df[f"old_{col}"].values[0] for col in cols], dtype=np.float32)
        z_list.append(old_state)
        
        # Then use new state from each row
        for col in cols:
            if col in df.columns:
                z_list.append(df[col].values)
        
        # Stack all trajectories
        z = np.column_stack(z_list[1:])  # Skip initial old_state since it's in z_list[0]
        # Actually, properly reconstruct
        z = np.vstack([z_list[0], np.array(z_list[1:]).T[1:]])
        
        trajectory = Trajectory(z=z, id=0, metadata={})
        trajectory.validate(dim)
        
        return metadata, [trajectory]


class DatasetConverter:
    """Convert between data formats."""
    
    @staticmethod
    def csv_to_standard_json(csv_path: str, 
                            json_path: str,
                            system_spec: 'SystemSpec',
                            dt: float = 0.1):
        """Convert CSV (old format) to standard JSON."""
        metadata, trajectories = StandardDatasetLoader.load_csv_trajectories(
            csv_path, system_spec, dt
        )
        
        # Serialize to JSON
        data = {
            "metadata": asdict(metadata),
            "trajectories": [
                {
                    "id": traj.id,
                    "z": traj.z.tolist(),
                    "t": traj.t.tolist() if traj.t is not None else None,
                    "metadata": traj.metadata,
                }
                for traj in trajectories
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Converted {csv_path} -> {json_path}")
    
    @staticmethod
    def trajectories_to_standard_json(trajectories: List[np.ndarray],
                                     system_spec: 'SystemSpec',
                                     dt: float,
                                     json_path: str,
                                     units: Dict[str, str] = None):
        """
        Create standard JSON from numpy trajectory arrays.
        
        Args:
            trajectories: List of arrays, each shape (steps, dimension)
            system_spec: System specification
            dt: Time step
            json_path: Output path
            units: Optional unit descriptions
        """
        if units is None:
            units = system_spec.units
        
        metadata = TrajectoryMetadata(
            system_name=system_spec.name,
            dimension=system_spec.dimension,
            dt=dt,
            num_trajectories=len(trajectories),
            trajectory_length=trajectories[0].shape[0] if trajectories else None,
            units=units,
        )
        
        traj_dicts = []
        for idx, z in enumerate(trajectories):
            t = np.arange(z.shape[0]) * dt
            traj_dicts.append({
                "id": idx,
                "z": z.astype(np.float64).tolist(),
                "t": t.astype(np.float64).tolist(),
                "metadata": {},
            })
        
        data = {
            "metadata": asdict(metadata),
            "trajectories": traj_dicts,
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Created {json_path} with {len(trajectories)} trajectories")


class StandardTrajectoryDataset:
    """
    Load trajectory data from standard format into PyTorch Dataset format.
    
    Compatible with TrajectoryDataset but from standard format.
    """
    
    def __init__(self, 
                 json_path: str,
                 system_spec: Optional['SystemSpec'] = None,
                 device: str = 'cpu',
                 no_data_to_gpu: bool = True):
        """
        Load from standard JSON format.
        
        Args:
            json_path: Path to standard JSON dataset
            system_spec: Optional SystemSpec to validate against
            device: PyTorch device
            no_data_to_gpu: If False, move data to GPU
        """
        from dpnn.system_spec import SystemSpec
        
        # Load data
        metadata, trajectories = StandardDatasetLoader.load_json(json_path)
        
        # Validate spec if provided
        if system_spec is not None:
            assert metadata.dimension == system_spec.dimension, \
                f"Spec dimension {system_spec.dimension} != data dimension {metadata.dimension}"
        
        self.metadata = metadata
        self.trajectories = trajectories
        self.device = device
        
        # Stack all trajectories and compute z_dot
        z_all_list = [traj.z for traj in trajectories]
        z_all_full = np.vstack(z_all_list)  # (total_steps, dim)
        
        # Compute z_dot via finite differences
        z_dot_all = (z_all_full[1:] - z_all_full[:-1]) / metadata.dt
        
        # Remove last state and get corresponding next states
        z_all = z_all_full[:-1]        # z(0), ..., z(N-1)
        z_all_next = z_all_full[1:]   # z(1), ..., z(N)
        
        # Convert to tensors
        self.z = torch.from_numpy(z_all).float()
        self.z_dot = torch.from_numpy(z_dot_all).float()
        
        # Compute midpoint: z_mid = 0.5*(z(t) + z(t+1))
        z_next_tensor = torch.from_numpy(z_all_next).float()
        self.z_mid = 0.5 * (self.z + z_next_tensor)
        
        # Move to device if requested
        if not no_data_to_gpu and device != 'cpu':
            self.z = self.z.to(device)
            self.z_dot = self.z_dot.to(device)
            self.z_mid = self.z_mid.to(device)
    
    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        """Return (z, z_dot, z_mid) for compatibility."""
        return (self.z[idx], self.z_dot[idx], self.z_mid[idx])
