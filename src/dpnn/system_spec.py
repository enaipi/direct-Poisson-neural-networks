"""
SystemSpec: Generic system description for arbitrary dynamical systems.

Replaces hardcoded model strings with self-describing system specifications.
Single source of truth for system dimensions and properties.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import torch


@dataclass
class SystemSpec:
    """
    Describes a general dynamical system for learning.
    
    This class enables generic learning on arbitrary systems without hardcoding
    dimension information or system-specific logic in the learning pipeline.
    """
    
    # Core identification
    name: str
    dimension: int
    
    # Physics configuration
    energy_parameterization: str = "neural"  # "neural", "quadratic", "hybrid"
    structure_tensor: str = "poisson"         # "poisson", "symplectic", "learned"
    
    # System properties
    conserved_quantities: List[str] = field(default_factory=list)
    symmetries: List[str] = field(default_factory=list)
    
    # Poisson structure (if not learned)
    # If structure_tensor == "poisson", this defines the bracket
    poisson_bracket_type: str = "canonical"  # "canonical", "rigid_body", "custom"
    
    # Metadata
    description: str = ""
    units: Dict[str, str] = field(default_factory=dict)
    
    # Optional: neural network structure preferences
    prefer_jacobian_loss: bool = False
    prefer_soft_jacobian: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "energy_parameterization": self.energy_parameterization,
            "structure_tensor": self.structure_tensor,
            "conserved_quantities": self.conserved_quantities,
            "symmetries": self.symmetries,
            "poisson_bracket_type": self.poisson_bracket_type,
            "description": self.description,
            "units": self.units,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemSpec":
        """Create from dictionary."""
        return cls(
            name=data.get("name"),
            dimension=data.get("dimension"),
            energy_parameterization=data.get("energy_parameterization", "neural"),
            structure_tensor=data.get("structure_tensor", "poisson"),
            conserved_quantities=data.get("conserved_quantities", []),
            symmetries=data.get("symmetries", []),
            poisson_bracket_type=data.get("poisson_bracket_type", "canonical"),
            description=data.get("description", ""),
            units=data.get("units", {}),
        )
    
    # ========================================================================
    # Registry: Predefined systems (single source of truth for dimensions)
    # ========================================================================
    
    @staticmethod
    def rigid_body() -> "SystemSpec":
        """RigidBody: 3D angular momentum state [mx, my, mz]."""
        return SystemSpec(
            name="RigidBody",
            dimension=3,
            energy_parameterization="quadratic",
            structure_tensor="poisson",
            poisson_bracket_type="rigid_body",
            conserved_quantities=["energy"],
            symmetries=["rotational"],
            description="3D rigid body motion with kinetic energy 0.5*(mx²/Ix + my²/Iy + mz²/Iz)",
            units={"state": "rad/s", "time": "s"},
        )
    
    @staticmethod
    def heavy_top() -> "SystemSpec":
        """HeavyTop: 6D state [mx, my, mz, rx, ry, rz]."""
        return SystemSpec(
            name="HeavyTop",
            dimension=6,
            energy_parameterization="hybrid",
            structure_tensor="poisson",
            poisson_bracket_type="rigid_body",
            conserved_quantities=["energy"],
            description="Rigid body with gravity: 6D phase space",
            units={"state": "mixed", "time": "s"},
        )
    
    @staticmethod
    def particle_3d() -> "SystemSpec":
        """Particle3D: 6D canonical phase space [rx, ry, rz, mx, my, mz]."""
        return SystemSpec(
            name="Particle3D",
            dimension=6,
            energy_parameterization="quadratic",
            structure_tensor="symplectic",
            poisson_bracket_type="canonical",
            conserved_quantities=["energy"],
            symmetries=["translation"],
            description="3D particle in canonical phase space",
            units={"position": "m", "momentum": "kg·m/s", "time": "s"},
        )
    
    @staticmethod
    def particle_2d() -> "SystemSpec":
        """Particle2D: 4D phase space [rx, ry, mx, my] with friction."""
        return SystemSpec(
            name="Particle2D",
            dimension=4,
            energy_parameterization="quadratic",
            structure_tensor="poisson",
            poisson_bracket_type="canonical",
            conserved_quantities=[],  # Energy not conserved due to friction
            description="2D particle with friction",
            units={"position": "m", "momentum": "kg·m/s", "time": "s"},
        )
    
    @staticmethod
    def shivamoggi_particle_nd() -> "SystemSpec":
        """Shivamoggi: 4D [u, x, y, z] system."""
        return SystemSpec(
            name="ShivamoggiParticleND",
            dimension=4,
            energy_parameterization="neural",
            structure_tensor="poisson",
            conserved_quantities=[],
            description="Shivamoggi N-dimensional particle system",
            units={"state": "mixed", "time": "s"},
        )
    
    @staticmethod
    def particle_nd(dimension: int) -> "SystemSpec":
        """Arbitrary N-dimensional canonical phase space."""
        return SystemSpec(
            name=f"ParticleND_D{dimension}",
            dimension=2 * dimension,
            energy_parameterization="neural",
            structure_tensor="symplectic",
            poisson_bracket_type="canonical",
            conserved_quantities=["energy"],
            description=f"{dimension}-dimensional particle in {2*dimension}D canonical phase space",
            units={"position": "m", "momentum": "kg·m/s", "time": "s"},
        )
    
    @staticmethod
    def custom(name: str, dimension: int, **kwargs) -> "SystemSpec":
        """Create custom system specification."""
        return SystemSpec(
            name=name,
            dimension=dimension,
            energy_parameterization=kwargs.get("energy_parameterization", "neural"),
            structure_tensor=kwargs.get("structure_tensor", "poisson"),
            conserved_quantities=kwargs.get("conserved_quantities", []),
            symmetries=kwargs.get("symmetries", []),
            poisson_bracket_type=kwargs.get("poisson_bracket_type", "canonical"),
            description=kwargs.get("description", ""),
            units=kwargs.get("units", {}),
            prefer_jacobian_loss=kwargs.get("prefer_jacobian_loss", False),
            prefer_soft_jacobian=kwargs.get("prefer_soft_jacobian", False),
        )


# Convenience factory to maintain backward compatibility
SYSTEM_REGISTRY = {
    "RB": SystemSpec.rigid_body(),
    "HT": SystemSpec.heavy_top(),
    "P3D": SystemSpec.particle_3d(),
    "K3D": SystemSpec.particle_3d(),  # Alias
    "P2D": SystemSpec.particle_2d(),
    "Sh": SystemSpec.shivamoggi_particle_nd(),
}


def get_system_spec(model_identifier: str, D: Optional[int] = None) -> SystemSpec:
    """
    Get system specification from model name or create ParticleND.
    
    Args:
        model_identifier: "RB", "HT", "P3D", "P2D", "Sh", or "D"
        D: Dimension parameter (required if model_identifier is "D")
    
    Returns:
        SystemSpec for the system
    """
    if model_identifier in SYSTEM_REGISTRY:
        return SYSTEM_REGISTRY[model_identifier]
    elif model_identifier == "D":
        if D is None:
            raise ValueError("Dimension D required for ParticleND system")
        return SystemSpec.particle_nd(D)
    else:
        raise ValueError(f"Unknown system: {model_identifier}")
