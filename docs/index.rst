Welcome to Direct Poisson Neural Networks's documentation!
==========================================================
Basic usage:

    python3 comparison.py --generate --steps=100 --implicit --soft --without --model=RB --folder_name=TEST

    python3 plot_compare.py --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST

The code is to be used by running first the comparison.py script, that simulates and learns dynamical systems, and then the plot_compare.py script, that shows the results.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   comparison
   plot_compare
   learn
   models
   simulate
   TrajectoryDataset

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
