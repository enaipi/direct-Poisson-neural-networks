Simulation (dpnn.simulation)
===========================

The simulation module contains trajectory simulation engines and CLI interface for generating trajectories with different models.

Simulator (simulator)
---------------------

Contains core simulation functionality including:
- ``simulate_batch()``: Simulate trajectories in batch
- ``save_simulation()``: Save simulation results
- Physics solvers for different mechanical systems

.. automodule:: dpnn.simulation.simulator
   :members:
   :undoc-members:
   :show-inheritance:

CLI Interface (simulate)
------------------------

Command-line interface for generating trajectories. Accessible via the ``simulate`` console command.

.. automodule:: dpnn.simulation.simulate
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Generate training dataset:

.. code-block:: bash

    simulate --generate --steps=50000 --model=RB

Generate test trajectories:

.. code-block:: bash

    simulate --steps=500 --generate
    simulate --steps=500 --implicit
    simulate --steps=500 --soft
    simulate --steps=500 --without

View available options:

.. code-block:: bash

    simulate --help

Supported Models
----------------

- **RB**: Rigid Body
- **HT**: Heavy Top
- **P3D**: Particle in Three Dimensions
- **P2D**: Particle in Two Dimensions
- **K3D**: Additional 3D model
