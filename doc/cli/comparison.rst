comparison - Generate and Compare Models
=========================================

The ``comparison`` command orchestrates the full workflow: it generates initial conditions, simulates ground truth trajectories, trains neural network models, simulates with learned models, and compares results.

This is the main entry point for the typical workflow.

.. automodule:: dpnn.comparison
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Run the complete comparison workflow:

.. code-block:: bash

    comparison --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST

Command Options
---------------

.. code-block:: bash

    comparison --help

Common Arguments
^^^^^^^^^^^^^^^^

- ``--generate``: Generate initial conditions and ground truth data
- ``--steps``: Number of time steps (default: 100)
- ``--model``: Mechanical system model (RB, HT, P3D, etc.; default: RB)
- ``--implicit``: Train implicit Jacobi model
- ``--soft``: Train soft Jacobi model
- ``--without``: Train model without Jacobi structure
- ``--folder_name``: Output folder name (default: TEST)

Example Workflows
-----------------

Complete workflow with all model variants:

.. code-block:: bash

    comparison --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST

Then visualize results:

.. code-block:: bash

    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Heavy top model:

.. code-block:: bash

    comparison --generate --steps=100 --implicit --soft --without --model=HT --folder_name=TEST_HT

Step-by-step Alternative
-------------------------

Instead of using ``comparison``, you can run each step manually:

.. code-block:: bash

    # 1. Generate training data
    simulate --generate --steps=50000 --model=RB
    
    # 2. Train models
    learn --method=without --model=RB
    learn --method=implicit --model=RB
    learn --method=soft --model=RB
    
    # 3. Generate test trajectories
    simulate --steps=500 --generate
    simulate --steps=500 --implicit
    simulate --steps=500 --soft
    simulate --steps=500 --without
    
    # 4. Visualize and compare
    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST
