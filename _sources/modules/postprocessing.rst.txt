Postprocessing (dpnn.postprocessing)
====================================

The postprocessing module contains visualization and analysis tools for comparing learned models with ground truth data.

Plot Compare (plot_compare)
---------------------------

CLI tool for visualizing and comparing trajectories from different models. Accessible via the ``plot-compare`` console command.

.. automodule:: dpnn.postprocessing.plot_compare
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Plot comparison of learned models with ground truth:

.. code-block:: bash

    plot-compare --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

Plot specific fields:

.. code-block:: bash

    plot-compare --plot_m --plot_E --plot_L

Plot energy surfaces:

.. code-block:: bash

    plot-compare --plot_Es --without --soft --implicit --folder_name=TEST

View available options:

.. code-block:: bash

    plot-compare --help

Features
--------

- Compare multiple models (without Jacobi, soft Jacobi, implicit Jacobi)
- Plot various fields (momentum, energy, angular momentum, etc.)
- Export figures to PNG
- Support for different mechanical systems (RB, HT, P3D, etc.)
- Error analysis and visualization
