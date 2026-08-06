"""
Shared helpers for the examples.
"""

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Dataset of consecutive states z_n -> z_{n+1} for residual training."""

    def __init__(self, trajectories, device=torch.device("cpu")):
        features = []
        targets = []
        for trajectory in trajectories:
            features.append(trajectory[:-1])
            targets.append(trajectory[1:])

        features = np.vstack(features).astype(np.float32)
        targets = np.vstack(targets).astype(np.float32)

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.mid = 0.5 * (self.features + self.targets)

        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.mid = self.mid.to(device)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.mid[idx]


def save_standard_json(
    trajectories,
    data_dir,
    filename,
    system_name,
    dt,
    dimension,
    units,
    state_order="q_then_p",
):
    """Save trajectories in the standard JSON format."""
    data = {
        "metadata": {
            "system_name": system_name,
            "dimension": dimension,
            "dt": dt,
            "num_trajectories": len(trajectories),
            "trajectory_length": len(trajectories[0]) if len(trajectories) > 0 else 0,
            "units": units,
        },
        "trajectories": [
            {
                "id": trajectory_id,
                "z": trajectory.tolist(),
                "t": (np.arange(len(trajectory)) * dt).tolist(),
                "metadata": {"state_order": state_order},
            }
            for trajectory_id, trajectory in enumerate(trajectories)
        ],
    }

    with open(data_dir / filename, "w") as f:
        json.dump(data, f, indent=2)


def save_transition_csv(trajectories, data_dir, filename, dt, component_dim, component_labels=None):
    """Save transition pairs for future legacy-style learner integration."""
    labels = component_labels or [str(i) for i in range(component_dim)]
    if len(labels) != component_dim:
        raise ValueError(f"Expected {component_dim} component labels, got {len(labels)}")

    rows = []
    for trajectory_id, trajectory in enumerate(trajectories):
        for step in range(len(trajectory) - 1):
            old_state = trajectory[step]
            new_state = trajectory[step + 1]
            row = {"trajectory_id": trajectory_id, "time": (step + 1) * dt}
            for idx, label in enumerate(labels):
                row[f"old_q{label}"] = old_state[idx]
                row[f"q{label}"] = new_state[idx]
            for idx, label in enumerate(labels):
                p_idx = component_dim + idx
                row[f"old_p{label}"] = old_state[p_idx]
                row[f"p{label}"] = new_state[p_idx]
            rows.append(row)

    pd.DataFrame(rows).to_csv(data_dir / filename, index=False)