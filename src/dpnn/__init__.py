"""Direct Poisson Neural Networks - A package for learning Poisson systems with neural networks."""

from dpnn.models.energy_nn import EnergyNet
from dpnn.models.tensor_nn import TensorNet, JacVectorNet
from dpnn.data.dataset import TrajectoryDataset
from dpnn.system_spec import SystemSpec, get_system_spec
from dpnn.training.general_learner import GeneralSystemLearner
from dpnn.training import create_learner

__version__ = "1.0.0"
__author__ = "Michal Sipka"

__all__ = [
    # Networks
    "EnergyNet",
    "TensorNet", 
    "JacVectorNet",
    # Data
    "TrajectoryDataset",
    # System specification
    "SystemSpec",
    "get_system_spec",
    # Learning
    "GeneralSystemLearner",
    "create_learner",
]

