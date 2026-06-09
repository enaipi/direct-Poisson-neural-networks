"""Direct Poisson Neural Networks - A package for learning Poisson systems with neural networks."""

from dpnn.models.energy_nn import EnergyNet
from dpnn.models.tensor_nn import TensorNet, JacVectorNet
from dpnn.data.dataset import TrajectoryDataset

__version__ = "1.0.0"
__author__ = "Michal Sipka"

__all__ = [
    "EnergyNet",
    "TensorNet", 
    "JacVectorNet",
    "TrajectoryDataset",
]
