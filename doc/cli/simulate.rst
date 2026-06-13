simulate - Generate Trajectories
================================

The ``simulate`` command generates trajectories for mechanical systems using either ground truth models or trained neural networks.

.. automodule:: dpnn.simulation.simulate
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Generate training dataset (ground truth):

.. code-block:: bash

    simulate --generate --steps=50000 --model=RB

Generate test trajectories with ground truth:

.. code-block:: bash

    simulate --steps=500 --generate

Generate trajectories with learned models:

.. code-block:: bash

    simulate --steps=500 --implicit
    simulate --steps=500 --soft
    simulate --steps=500 --without

Command Options
---------------

.. code-block:: bash

    simulate --help

Common Arguments
^^^^^^^^^^^^^^^^

- ``--generate``: Generate and use ground truth dynamics
- ``--implicit``: Use implicit Jacobi trained model
- ``--soft``: Use soft Jacobi trained model
- ``--without``: Use model without Jacobi structure
- ``--steps``: Number of time steps to simulate (default: 500)
- ``--model``: Mechanical system model (RB, HT, P3D, etc.; default: RB)
- ``--folder_name``: Output folder name (default: TEST)

Example Workflows
-----------------

Generate large training dataset:

.. code-block:: bash

    simulate --generate --steps=50000 --model=RB --folder_name=TEST

Generate test trajectories with all model variants:

.. code-block:: bash

    simulate --steps=500 --generate --folder_name=TEST
    simulate --steps=500 --implicit --folder_name=TEST
    simulate --steps=500 --soft --folder_name=TEST
    simulate --steps=500 --without --folder_name=TEST

Output
------

Trajectories are saved to ``{folder_name}/data/`` directory:

- ``dataset.xyz``: Training dataset (if ``--generate`` with large steps)
- ``generalization.xyz``: Ground truth test trajectories (if ``--generate``)
- ``learned_implicit.xyz``: Trajectories from implicit model
- ``learned_soft.xyz``: Trajectories from soft model
- ``learned_without.xyz``: Trajectories from without model

Supported Models
----------------

- **RB**: Rigid Body
- **HT**: Heavy Top
- **P3D**: Particle in Three Dimensions
- **P2D**: Particle in Two Dimensions
- **K3D**: Additional 3D model

Typical Workflow
----------------

.. code-block:: bash

    # 1. Generate training data
    simulate --generate --steps=50000 --model=RB
    
    # 2. Train models (see: learn command)
    learn --method=without --model=RB
    learn --method=implicit --model=RB
    learn --method=soft --model=RB
    
    # 3. Generate test trajectories
    simulate --steps=500 --generate
    simulate --steps=500 --implicit
    simulate --steps=500 --soft
    simulate --steps=500 --without
    
    # 4. Compare results
    plot-compare --plot_RB_errors --GT --without --implicit --soft
