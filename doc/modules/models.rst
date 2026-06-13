Models (dpnn.models)
====================

The models module contains neural network architectures for learning energy functions and Poisson bracket tensor structures, along with physics solvers for different mechanical systems.

Submodules
----------

Energy Networks (energy_nn)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains the EnergyNet class that learns energy functions of dynamical systems.

.. automodule:: dpnn.models.energy_nn
   :members:
   :undoc-members:
   :show-inheritance:

Tensor Networks (tensor_nn)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contains neural networks for learning Poisson bracket tensor structures:
- TensorNet: Learns the Poisson bracket tensor
- JacVectorNet: Learns Jacobian vector fields

.. automodule:: dpnn.models.tensor_nn
   :members:
   :undoc-members:
   :show-inheritance:

Physical Models (physical_models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The physical_models subpackage contains 21 physics solver classes organized across 7 modules for simulating Hamiltonian systems with different integration methods (Crank-Nicolson, IMR, RK4, Forward Euler, etc.) and model types (rigid body, heavy top, particles, etc.).

**Base Module (base)**
- RigidBody: Base class for rigid body systems
- load_models: Load neural network models for physics solvers

.. automodule:: dpnn.models.physical_models.base
   :members:
   :undoc-members:
   :show-inheritance:

**Rigid Body Integrators (rigid_body_integrators)**

Five integrator variants for RigidBody systems:
- RBEhrenfest: Ehrenfest integrator
- RBESeReCN: Energy self-regularized Crank-Nicolson
- RBIMR: Implicit Midpoint Rule (2nd order)
- RBRK4: 4th order Runge-Kutta
- RBESeReFE: Self-regularized Forward Euler

.. automodule:: dpnn.models.physical_models.rigid_body_integrators
   :members:
   :undoc-members:
   :show-inheritance:

**Neural Solvers (neural)**

Physics solvers with neural network energy/Poisson structures:
- Neural: RigidBody with neural networks
- RBNeuralIMR: Neural RigidBody with Implicit Midpoint Rule

.. automodule:: dpnn.models.physical_models.neural
   :members:
   :undoc-members:
   :show-inheritance:

**Heavy Top Models (heavy_top)**

Models for 6D heavy top systems (3D angular momentum + 3D orientation):
- HeavyTopCN: Crank-Nicolson integrator
- HeavyTopIMR: Implicit Midpoint Rule integrator
- HeavyTopNeural: Neural network variant
- HeavyTopNeuralIMR: Neural with Implicit Midpoint Rule

.. automodule:: dpnn.models.physical_models.heavy_top
   :members:
   :undoc-members:
   :show-inheritance:

**3D Particle Models (particle_3d)**

Models for 3D particles (6D state: 3D position + 3D momentum):
- Particle3DCN: Crank-Nicolson integrator
- Particle3DIMR: Implicit Midpoint Rule integrator
- Particle3DNeural: Neural network variant
- Particle3DNeuralIMR: Neural with Implicit Midpoint Rule
- Particle3DKeplerIMR: Kepler potential variant (1/r²)

.. automodule:: dpnn.models.physical_models.particle_3d
   :members:
   :undoc-members:
   :show-inheritance:

**2D Particle Models (particle_2d)**

Models for 2D particles with friction (4D state: 2D position + 2D momentum):
- Particle2DIMR: Implicit Midpoint Rule with dissipation
- Particle2DNeural: Neural network variant

.. automodule:: dpnn.models.physical_models.particle_2d
   :members:
   :undoc-members:
   :show-inheritance:

**Specialized Systems (shivamoggi_particlend)**

Shivamoggi 4D system and N-dimensional oscillators:
- ShivamoggiIMR: Implicit Midpoint Rule for Shivamoggi system
- ShivamoggiNeural: Neural variant of Shivamoggi system
- ParticleNDCN: Generic N-dimensional harmonic oscillator (Crank-Nicolson)
- ParticleNDCNNeural: N-dimensional oscillator with neural networks

.. automodule:: dpnn.models.physical_models.shivamoggi_particlend
   :members:
   :undoc-members:
   :show-inheritance:

**Package API (physical_models.__init__)**

The physical_models subpackage provides convenient access to all 21 classes and utilities:

.. automodule:: dpnn.models.physical_models
   :members:
   :undoc-members:
   :show-inheritance:

Backward Compatibility (rigid_body)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rigid_body module serves as a backward compatibility shim, re-exporting all 21 classes from the reorganized physical_models subpackage. Code using the old import path continues to work without modification:

.. code-block:: python

   # Old import path (still works)
   from dpnn.models.rigid_body import RigidBody, RBIMR

   # New preferred path (recommended)
   from dpnn.models.physical_models import RigidBody, RBIMR

.. automodule:: dpnn.models.rigid_body
   :members:
   :undoc-members:
   :show-inheritance:

Legacy Energy Module (energy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Original energy module (deprecated - use energy_nn and tensor_nn instead).

.. automodule:: dpnn.models.energy
   :members:
   :undoc-members:
   :show-inheritance:

Package-level API
-----------------

The models package exports the main neural network classes for convenient access:

.. automodule:: dpnn.models
   :members:
   :undoc-members:
   :show-inheritance:
