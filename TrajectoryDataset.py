from torch.utils.data import Dataset
import numpy as np
import torch

class TrajectoryDataset(Dataset):
    """TRAJECTORY DATASET"""

    def __init__(self, dataframe, model = "RB", device=None):
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
        else:
            raise Exception("Unknown model.")

        mid = 0.5 * (features + targets)
        
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.mid = torch.from_numpy(mid)
        
        if device is not None:
            self.features = self.features.to(device)
            self.targets = self.targets.to(device)
            self.mid = self.mid.to(device)
    
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return (self.features[idx], self.targets[idx], self.mid[idx])
