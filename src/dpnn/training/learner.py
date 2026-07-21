"""
DEPRECATED: Learner is deprecated. Use RobustLearner instead.

This module provides backward compatibility by wrapping RobustLearner.
All functionality has been moved to RobustLearner in robust_learner.py.

Migration path:
    from dpnn.training.learner import Learner
    learner = Learner("RB", neurons=64)
    
Should become:
    from dpnn.training.robust_learner import RobustLearner
    learner = RobustLearner(model="RB", neurons=64)

Or use as thin wrapper (still works):
    from dpnn.training.learner import Learner
    learner = Learner("RB", neurons=64)  # Still works, just warns
"""

import warnings
import argparse
import os

from dpnn.training.robust_learner import RobustLearner


# Deprecation warning module
def _deprecation_warning():
    warnings.warn(
        "Learner is deprecated and will be removed in a future version. "
        "Please use RobustLearner from dpnn.training.robust_learner instead.",
        DeprecationWarning,
        stacklevel=3
    )


# Forward all exports from RobustLearner
__all__ = ['Learner', 'LearnerIMR', 'LearnerRK4', 'check_folder']


# Default parameters (for backward compatibility)
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


class Learner(RobustLearner):
    """
    Backward-compatible Learner class.
    
    This is a thin wrapper around RobustLearner for backward compatibility.
    
    DEPRECATED: Use RobustLearner instead.
    
    Example:
        # Old way (still works):
        learner = Learner(model="RB", neurons=64, batch_size=32)
        learner.learn(method="soft", epochs=10)
        
        # New way (recommended):
        learner = RobustLearner(model="RB", neurons=64, batch_size=32)
        learner.learn(method="soft", epochs=10)  # Same API
    """
    
    def __init__(self, model, **kwargs):
        """Initialize Learner with deprecation warning."""
        _deprecation_warning()
        super().__init__(model=model, **kwargs)


class LearnerIMR(RobustLearner):
    """
    Backward-compatible LearnerIMR subclass.
    
    Uses implicit midpoint rule integration scheme.
    
    DEPRECATED: Use RobustLearner(model=..., integration_scheme="imr") instead.
    """
    
    def __init__(self, model, **kwargs):
        """Initialize LearnerIMR with IMR scheme."""
        _deprecation_warning()
        kwargs['integration_scheme'] = 'imr'
        super().__init__(model=model, **kwargs)


class LearnerRK4(RobustLearner):
    """
    Backward-compatible LearnerRK4 subclass.
    
    Uses Runge-Kutta 4th-order integration scheme.
    
    DEPRECATED: Use RobustLearner(model=..., integration_scheme="rk4") instead.
    """
    
    def __init__(self, model, **kwargs):
        """Initialize LearnerRK4 with RK4 scheme."""
        _deprecation_warning()
        kwargs['integration_scheme'] = 'rk4'
        super().__init__(model=model, **kwargs)


def check_folder(name):
    """
    Check if folder exists, create if not.
    
    Creates data/ and saved_models/ subdirectories.
    """
    print("Checking folder ", name)
    name = os.getcwd() + "/" + name
    data_name = name + "/data"
    models_name = name + "/saved_models"
    if not os.path.exists(data_name):
        print("Making folder: " + data_name)
        os.makedirs(data_name)
    if not os.path.exists(models_name):
        print("Making folder: " + models_name)
        os.makedirs(models_name)


if __name__ == "__main__":
    # Command-line interface (backward compatible)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=DEFAULT_epochs, type=int, help="Number of epochs")
    parser.add_argument("--neurons", default=DEFAULT_neurons, type=int, help="Number of hidden neurons")
    parser.add_argument("--layers", default=DEFAULT_layers, type=int, help="Number of layers")
    parser.add_argument("--batch_size", default=DEFAULT_batch_size, type=int, help="Batch size")
    parser.add_argument("--dt", default=DEFAULT_dt, type=float, help="Step size")
    parser.add_argument("--prefactor", default=DEFAULT_prefactor, type=float, help="Prefactor in the loss")
    parser.add_argument("--learning_rate", default=DEFAULT_learning_rate, type=float, help="Learning rate")
    parser.add_argument("--model", type=str, help="Model = RB, HT, or P3D.", required=True)
    parser.add_argument("--name", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--method", default="without", type=str, help="Method: without, implicit, or soft")
    parser.add_argument("--jacobi_loss_mode", default=DEFAULT_jacobi_loss_mode, type=str,
                        choices=["exact", "hutchinson", "spectral", "manual", "exact_backward", "hutchinson_batch"],
                        help="Jacobi loss evaluation mode")
    parser.add_argument("--hutchinson_samples", default=DEFAULT_hutchinson_samples, type=int,
                        help="Number of Hutchinson probe vectors for Jacobi loss")

    args = parser.parse_args()
    check_folder(args.name)

    learner = Learner(
        args.model,
        neurons=args.neurons,
        layers=args.layers,
        batch_size=args.batch_size,
        dt=args.dt,
        name=args.name,
        jacobi_loss_mode=args.jacobi_loss_mode,
        hutchinson_samples=args.hutchinson_samples
    )
    learner.learn(
        method=args.method,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        prefactor=args.prefactor
    )
