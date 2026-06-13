Data (dpnn.data)
================

The data module contains utilities for loading and managing trajectory datasets.

TrajectoryDataset
-----------------

PyTorch Dataset class for loading trajectory data from CSV files.

.. automodule:: dpnn.data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

.. code-block:: python

    from dpnn.data import TrajectoryDataset
    
    # Load trajectory data
    dataset = TrajectoryDataset('path/to/data.xyz')
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=32)
