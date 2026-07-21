"""Training utilities and learner classes."""

from .learner import (
    Learner,
    LearnerIMR,
    LearnerRK4,
    check_folder,
    DEFAULT_folder_name,
    DEFAULT_dataset,
    DEFAULT_jacobi_loss_mode,
    DEFAULT_hutchinson_samples,
)

from .general_learner import GeneralSystemLearner

# Backward compatibility factory
from dpnn.system_spec import get_system_spec

def create_learner(model_identifier: str, **kwargs):
    """
    Backward compatible factory for creating learners.
    
    Supports both old API (model strings) and new API (SystemSpec).
    
    Examples:
        # Old API (still works)
        learner = create_learner("RB", batch_size=32)
        
        # New API
        spec = SystemSpec.rigid_body()
        learner = GeneralSystemLearner(spec, batch_size=32)
    """
    import warnings
    D = kwargs.pop("D", None)
    
    # Get system spec from model identifier
    spec = get_system_spec(model_identifier, D=D)
    
    # Create learner with remaining kwargs
    return GeneralSystemLearner(spec, **kwargs)


__all__ = [
    "Learner",
    "LearnerIMR",
    "LearnerRK4",
    "GeneralSystemLearner",
    "create_learner",
    "check_folder",
    "DEFAULT_folder_name",
    "DEFAULT_dataset",
    "DEFAULT_jacobi_loss_mode",
    "DEFAULT_hutchinson_samples",
]

