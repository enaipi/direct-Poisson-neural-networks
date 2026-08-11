Direct Poisson Neural Networks
==============================

A Python package for learning non-symplectic mechanical systems using neural networks based on direct Poisson bracket structures.

Installation
============

Install the package in development mode with documentation dependencies:

.. code-block:: bash

    pip install -e ".[docs]"

Quick Start
===========

The typical workflow consists of:

1. **Generate and learn**: Run the comparison workflow which simulates initial conditions, learns dynamics, simulates with learned models, and compares results:

   .. code-block:: bash

       comparison --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST
       plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

2. **Alternative step-by-step approach**:

   Generate training dataset:

   .. code-block:: bash

       simulate --generate --steps=50000 --model=RB

   Train models:

   .. code-block:: bash

       learn --method=without --model=RB
       learn --method=implicit --model=RB
       learn --method=soft --model=RB

   Generate test trajectories with learned models:

   .. code-block:: bash

       simulate --steps=500 --generate
       simulate --steps=500 --implicit
       simulate --steps=500 --soft
       simulate --steps=500 --without

   Visualize results:

   .. code-block:: bash

       plot-compare --plot_m --plot_E --plot_L

Available Models
================

- **RB**: Rigid Body
- **HT**: Heavy Top
- **P3D**: Particle in Three Dimensions
- **P2D**: Particle in Two Dimensions (Shivamoggi coordinates)
- **K3D**: Additional 3D model

Package Structure
=================

The package is organized into the following modules:

.. toctree::
   :maxdepth: 2
   :caption: Core Modules:

   modules/models
   modules/training
   modules/simulation
   modules/data
   modules/postprocessing

.. toctree::
   :maxdepth: 2
   :caption: CLI Reference:

   cli/comparison
   cli/plot_compare
   cli/learn
   cli/simulate

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
