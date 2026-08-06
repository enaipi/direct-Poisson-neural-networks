"""Rigid body and particle models with various integrators and neural network support."""

from .base import (
    load_models,
    GeneralSystem,
)

from .rigid_body import (
    RigidBody,
)

from .rigid_body_integrators import (
    RBEhrenfest,
    RBESeReCN,
    RBIMR,
    RBRK4,
    RBESeReFE,
)

from .neural import (
    Neural,
    RBNeuralIMR,
)

from .heavy_top import (
    HeavyTopCN,
    HeavyTopIMR,
    HeavyTopNeural,
    HeavyTopNeuralIMR,
)

from .particle_3d import (
    Particle3DCN,
    Particle3DIMR,
    Particle3DNeural,
    Particle3DNeuralIMR,
    Particle3DKeplerIMR,
)

from .particle_2d import (
    Particle2DIMR,
    Particle2DNeural,
)

from .fpu import (
    FPUCN,
    FPUIMR,
    FPUNeural,
    FPUNeuralIMR,
)

from .shivamoggi_particlend import (
    ShivamoggiIMR,
    ShivamoggiNeural,
    ParticleNDCN,
    ParticleNDCNNeural,
)

from .harmonic import (
    HarmonicCN,
    HarmonicIMR,
    HarmonicNeural,
    HarmonicNeuralIMR
)

__all__ = [
    # Generic infrastructure
    "GeneralSystem",
    "load_models",
    "RigidBody",
    
    # Rigid Body integrators
    "RBEhrenfest",
    "RBESeReCN",
    "RBIMR",
    "RBRK4",
    "RBESeReFE",
    
    # Rigid Body with neural networks
    "Neural",
    "RBNeuralIMR",
    
    # Heavy Top models
    "HeavyTopCN",
    "HeavyTopIMR",
    "HeavyTopNeural",
    "HeavyTopNeuralIMR",
    
    # 3D Particle models
    "Particle3DCN",
    "Particle3DIMR",
    "Particle3DNeural",
    "Particle3DNeuralIMR",
    "Particle3DKeplerIMR",
    
    # 2D Particle models
    "Particle2DIMR",
    "Particle2DNeural",

    # Fermi-Pasta-Ulam models
    "FPUCN",
    "FPUIMR",
    "FPUNeural",
    "FPUNeuralIMR",
    
    # Shivamoggi and N-D particle models
    "ShivamoggiIMR",
    "ShivamoggiNeural",
    "ParticleNDCN",
    "ParticleNDCNNeural",

    # Harmonic particles
    "HarmonicCN",
    "HarmonicIMR",
    "HarmonicNeural",
    "HarmonicNeuralIMR"
]
