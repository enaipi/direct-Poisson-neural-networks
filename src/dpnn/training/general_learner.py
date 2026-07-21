"""
DEPRECATED: GeneralSystemLearner is deprecated. Use RobustLearner instead.

This module provides backward compatibility for the SystemSpec-based API.
All functionality has been moved to RobustLearner in robust_learner.py.

Migration path:
    from dpnn.training.general_learner import GeneralSystemLearner
    learner = GeneralSystemLearner(system_spec=spec)
    
Should become:
    from dpnn.training.robust_learner import RobustLearner
    learner = RobustLearner(system_spec=spec)

Or use as thin wrapper (still works):
    from dpnn.training.general_learner import GeneralSystemLearner
    learner = GeneralSystemLearner(system_spec=spec)  # Still works, just warns
"""

import warnings
from typing import Optional

from dpnn.system_spec import SystemSpec
from dpnn.training.robust_learner import RobustLearner


def _deprecation_warning():
    """Show deprecation warning."""
    warnings.warn(
        "GeneralSystemLearner is deprecated and will be removed in a future version. "
        "Please use RobustLearner from dpnn.training.robust_learner instead.",
        DeprecationWarning,
        stacklevel=3
    )


__all__ = ['GeneralSystemLearner']


class GeneralSystemLearner(RobustLearner):
    """
    Backward-compatible GeneralSystemLearner class.
    
    This is a thin wrapper around RobustLearner for backward compatibility
    with code using the SystemSpec-based API.
    
    DEPRECATED: Use RobustLearner instead.
    
    Example:
        # Old way (still works):
        learner = GeneralSystemLearner(system_spec=spec, neurons=64, batch_size=32)
        learner.learn(method="soft", epochs=10)
        
        # New way (recommended):
        learner = RobustLearner(system_spec=spec, neurons=64, batch_size=32)
        learner.learn(method="soft", epochs=10)  # Same API
    """
    
    def __init__(self,
                 system_spec: SystemSpec,
                 batch_size: int = 32,
                 neurons: int = 64,
                 layers: int = 2,
                 device: str = 'cpu',
                 dropout_rate: float = 0.0,
                 quad_features: bool = False,
                 jacobi_loss_mode: str = "exact",
                 hutchinson_samples: int = 3,
                 **kwargs):
        """
        Initialize GeneralSystemLearner.
        
        Args:
            system_spec: SystemSpec describing the system
            batch_size: Training batch size
            neurons: Neurons per layer in networks
            layers: Number of layers
            device: PyTorch device ('cpu' or 'cuda')
            dropout_rate: Dropout rate in networks
            quad_features: Add quadratic features to energy net
            jacobi_loss_mode: Jacobi loss evaluation mode
            hutchinson_samples: Samples for stochastic estimation
            **kwargs: Additional arguments for RobustLearner
        """
        _deprecation_warning()
        
        # Pass SystemSpec-based configuration to RobustLearner
        super().__init__(
            system_spec=system_spec,
            batch_size=batch_size,
            neurons=neurons,
            layers=layers,
            device=device,
            dropout_rate=dropout_rate,
            quad_features=quad_features,
            jacobi_loss_mode=jacobi_loss_mode,
            hutchinson_samples=hutchinson_samples,
            **kwargs
        )
