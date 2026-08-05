"""Training utilities and learner classes for Hamiltonian/Poisson systems.

This module provides the HamiltonianLearner class for learning dynamical systems
from trajectory data. It supports both generated and external data, works with multiple
physical systems (RigidBody, HeavyTop, Particle systems, etc.), and provides flexible
training methods and Jacobi loss variants.

Quick Start:
    from dpnn.training import HamiltonianLearner
    learner = HamiltonianLearner(model="RB", neurons=64, batch_size=32)
    learner.learn(method="soft", epochs=10, learning_rate=0.0001)
    
    # With external data
    learner = HamiltonianLearner(model="RB", external_data_path="data.csv")
    learner.learn(method="implicit", epochs=20)
    
    # With SystemSpec for arbitrary systems
    from dpnn.system_spec import SystemSpec
    spec = SystemSpec.rigid_body()
    learner = HamiltonianLearner(system_spec=spec, neurons=128)
    learner.learn(method="soft")
"""

from .hamiltonian_learner import HamiltonianLearner

# Default parameters
DEFAULT_dataset = "data/dataset.xyz"
DEFAULT_batch_size = 32
DEFAULT_dt = 0.1 
DEFAULT_learning_rate = 1.0e-05
DEFAULT_epochs = 10 
DEFAULT_prefactor = 1.0
DEFAULT_jacobi_prefactor = 1.0
DEFAULT_neurons = 64
DEFAULT_layers = 2
DEFAULT_folder_name = "."
DEFAULT_jacobi_loss_mode = "exact"
DEFAULT_hutchinson_samples = 3

__all__ = [
    'HamiltonianLearner',
    'DEFAULT_dataset',
    'DEFAULT_batch_size',
    'DEFAULT_dt',
    'DEFAULT_learning_rate',
    'DEFAULT_epochs',
    'DEFAULT_prefactor',
    'DEFAULT_jacobi_prefactor',
    'DEFAULT_neurons',
    'DEFAULT_layers',
    'DEFAULT_folder_name',
    'DEFAULT_jacobi_loss_mode',
    'DEFAULT_hutchinson_samples',
]

