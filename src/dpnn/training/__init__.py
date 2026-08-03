"""Training utilities and learner classes for Hamiltonian/Poisson systems.

This module provides the main HamiltonianLearner class for learning dynamical systems
from trajectory data. It supports both generated and external data, works with multiple
physical systems (RigidBody, HeavyTop, Particle systems, etc.), and provides flexible
training methods and Jacobi loss variants.

Quick Start:
    # Basic usage with generated data
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

Backward Compatibility:
    Old code using Learner, LearnerIMR, etc. still works:
        from dpnn.training import Learner  # Works, shows deprecation warning
        learner = Learner(model="RB")
    
    But we recommend updating to HamiltonianLearner for all new code.
"""

# Main public API
from .hamiltonian_learner import HamiltonianLearner

# Backward compatibility - these show deprecation warnings
from ._legacy import (
    Learner,
    LearnerIMR,
    LearnerRK4,
    check_folder,
    DEFAULT_dataset,
    DEFAULT_batch_size,
    DEFAULT_dt,
    DEFAULT_learning_rate,
    DEFAULT_epochs,
    DEFAULT_prefactor,
    DEFAULT_jacobi_prefactor,
    DEFAULT_neurons,
    DEFAULT_layers,
    DEFAULT_folder_name,
    DEFAULT_jacobi_loss_mode,
    DEFAULT_hutchinson_samples,
)

# Also export GeneralSystemLearner for backward compatibility
from ._legacy_general import GeneralSystemLearner

__all__ = [
    # Main API (recommended)
    'HamiltonianLearner',
    
    # Legacy wrappers (deprecated, for backward compatibility)
    'Learner',
    'LearnerIMR',
    'LearnerRK4',
    'GeneralSystemLearner',
    'check_folder',
    
    # Default parameters (for backward compatibility)
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
    "DEFAULT_dataset",
    "DEFAULT_jacobi_loss_mode",
    "DEFAULT_hutchinson_samples",
]

