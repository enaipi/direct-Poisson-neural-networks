learn - Train Neural Network Models
===================================

The ``learn`` command trains neural network models to learn the dynamics of mechanical systems from trajectory data.

.. automodule:: dpnn.training.learn
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Train a model with a specific method:

.. code-block:: bash

    learn --method=without --model=RB
    learn --method=implicit --model=RB
    learn --method=soft --model=RB

Command Options
---------------

.. code-block:: bash

    learn --help

Common Arguments
^^^^^^^^^^^^^^^^

- ``--method``: Training method (without, implicit, soft)
- ``--model``: Mechanical system model (RB, HT, P3D, etc.; default: RB)
- ``--folder_name``: Output folder name (default: TEST)
- Additional hyperparameters and training options

Training Methods
----------------

**without**: 
  Train a model without any Jacobi structure constraint

**implicit**: 
  Train with implicit Jacobi correction method

**soft**: 
  Train with soft Jacobi structure constraint

Example Workflows
-----------------

Train all three model variants:

.. code-block:: bash

    learn --method=without --model=RB
    learn --method=implicit --model=RB
    learn --method=soft --model=RB

Train for different mechanical systems:

.. code-block:: bash

    learn --method=implicit --model=HT
    learn --method=soft --model=P3D

Output
------

Trained models are saved to ``{folder_name}/saved_models/`` directory:

- ``{method}_jacobi_energy``: Learned energy function network
- ``{method}_jacobi_L`` or ``{method}_jacobi_J``: Learned Poisson structure

Training Data
-------------

The command expects pre-generated trajectory data in ``{folder_name}/data/dataset.xyz``.

Generate training data using:

.. code-block:: bash

    simulate --generate --steps=50000 --model=RB --folder_name=TEST
