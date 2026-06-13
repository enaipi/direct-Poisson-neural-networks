plot-compare - Visualize and Compare Models
===========================================

The ``plot-compare`` command generates visualizations comparing ground truth trajectories with learned model trajectories.

.. automodule:: dpnn.postprocessing.plot_compare
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

Compare models with ground truth:

.. code-block:: bash

    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Plot specific fields:

.. code-block:: bash

    plot-compare --plot_m --plot_E --plot_L --folder_name=TEST

Command Options
---------------

.. code-block:: bash

    plot-compare --help

Common Arguments
^^^^^^^^^^^^^^^^

- ``--GT``: Include ground truth data in plots
- ``--without``: Include model without Jacobi structure
- ``--implicit``: Include implicit Jacobi model
- ``--soft``: Include soft Jacobi model
- ``--dataset``: Include dataset trajectories
- ``--folder_name``: Output folder name (default: TEST)
- ``--export``: Save figures to PNG files
- ``--model``: Mechanical system model (RB, HT, P3D, etc.; default: RB)

Plotting Options
^^^^^^^^^^^^^^^^

**Field Plotting:**

- ``--plot_m``: Plot momentum (m_x, m_y, m_z)
- ``--plot_r``: Plot position (r_x, r_y, r_z)
- ``--plot_E``: Plot energy
- ``--plot_L``: Plot Poisson bivector components
- ``--plot_field FIELD``: Plot custom field

**Error Analysis:**

- ``--plot_RB_errors``: Plot momentum errors (for Rigid Body)
- ``--plot_HT_errors``: Plot errors for Heavy Top
- ``--plot_P3D_errors``: Plot errors for 3D particle
- ``--plot_P2D_errors``: Plot errors for 2D particle (Shivamoggi)
- ``--plot_L_errors``: Plot Poisson bracket tensor errors
- ``--plot_msq_errors``: Plot squared momentum magnitude errors
- ``--plot_mrs_errors``: Plot m·r errors
- ``--plot_am_errors``: Plot angular momentum errors
- ``--plot_training_errors``: Plot training/validation loss comparison

**Advanced Plotting:**

- ``--plot_Es``: Plot energy surfaces
- ``--plot_Ls``: Plot Poisson tensor surfaces
- ``--plot_Jacobi``: Plot Jacobian information
- ``--plot_Casimir``: Plot Casimir functions
- ``--plot_spectrum_errors``: Plot spectrum errors
- ``--plot_compatibility``: Plot compatibility errors

Example Workflows
-----------------

Basic comparison - all models with ground truth:

.. code-block:: bash

    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Plot specific fields:

.. code-block:: bash

    plot-compare --plot_m --plot_E --plot_L --GT --soft --implicit --without

Plot energy surfaces and tensors:

.. code-block:: bash

    plot-compare --plot_Es --plot_Ls --soft --implicit --without

Export figures:

.. code-block:: bash

    plot-compare --export --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Compare different models:

.. code-block:: bash

    # Heavy Top
    plot-compare --plot_HT_errors --GT --soft --implicit --model=HT --folder_name=TEST_HT
    
    # 3D Particle
    plot-compare --plot_P3D_errors --GT --soft --implicit --model=P3D --folder_name=TEST_P3D

Output
------

When using ``--export``, figures are saved to ``{folder_name}/`` directory as PNG files.

Typical Workflow
----------------

.. code-block:: bash

    # 1. Generate data and train models (see: comparison command)
    comparison --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST
    
    # 2. Visualize results
    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST
    
    # 3. Generate additional plots
    plot-compare --plot_m --plot_E --plot_L --folder_name=TEST
    plot-compare --plot_Es --plot_Ls --folder_name=TEST
