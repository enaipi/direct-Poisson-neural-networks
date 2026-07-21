"""
Backward compatibility shim: imports from the reorganized physical_models package.

All classes are now organized in the physical_models/ subpackage.
This module allows old code using 'from dpnn.models.rigid_body import *' to continue working.

New code should prefer:
    from dpnn.models.physical_models import RigidBody, load_models
"""

# Import and re-export all classes from the new physical_models package
from dpnn.models.physical_models.base import load_models
from dpnn.models.physical_models.rigid_body import RigidBody
from dpnn.models.physical_models.rigid_body_integrators import (
    RBEhrenfest, RBESeReCN, RBIMR, RBRK4, RBESeReFE
)
from dpnn.models.physical_models.neural import Neural, RBNeuralIMR
from dpnn.models.physical_models.heavy_top import (
    HeavyTopCN, HeavyTopIMR, HeavyTopNeural, HeavyTopNeuralIMR
)
from dpnn.models.physical_models.particle_3d import (
    Particle3DCN, Particle3DIMR, Particle3DNeural, Particle3DNeuralIMR, Particle3DKeplerIMR
)
from dpnn.models.physical_models.particle_2d import Particle2DIMR, Particle2DNeural
from dpnn.models.physical_models.shivamoggi_particlend import (
    ShivamoggiIMR, ShivamoggiNeural, ParticleNDCN, ParticleNDCNNeural
)

__all__ = [
    "load_models", "RigidBody",
    "RBEhrenfest", "RBESeReCN", "RBIMR", "RBRK4", "RBESeReFE",
    "Neural", "RBNeuralIMR",
    "HeavyTopCN", "HeavyTopIMR", "HeavyTopNeural", "HeavyTopNeuralIMR",
    "Particle3DCN", "Particle3DIMR", "Particle3DNeural", "Particle3DNeuralIMR", "Particle3DKeplerIMR",
    "Particle2DIMR", "Particle2DNeural",
    "ShivamoggiIMR", "ShivamoggiNeural", "ParticleNDCN", "ParticleNDCNNeural",
]

__all__ = [
    "load_models", "RigidBody",
    "RBEhrenfest", "RBESeReCN", "RBIMR", "RBRK4", "RBESeReFE",
    "Neural", "RBNeuralIMR",
    "HeavyTopCN", "HeavyTopIMR", "HeavyTopNeural", "HeavyTopNeuralIMR",
    "Particle3DCN", "Particle3DIMR", "Particle3DNeural", "Particle3DNeuralIMR", "Particle3DKeplerIMR",
    "Particle2DIMR", "Particle2DNeural",
    "ShivamoggiIMR", "ShivamoggiNeural", "ParticleNDCN", "ParticleNDCNNeural",
]
