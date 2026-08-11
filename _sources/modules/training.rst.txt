Training (dpnn.training)
========================

The training module contains the neural network learner classes and CLI interface for training energy and tensor network models on trajectory data.

Learner Classes (learner)
-------------------------

Contains the Learner class hierarchy for training different types of models:
- **Learner**: Base learner class
- **LearnerIMR**: Learner for implicit method with Jacobi correction
- **LearnerRK4**: Learner with RK4 integrator

.. automodule:: dpnn.training.learner
   :members:
   :undoc-members:
   :show-inheritance:

CLI Interface (learn)
---------------------

Command-line interface for training models. Accessible via the ``learn`` console command.

.. automodule:: dpnn.training.learn
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Train a model using the CLI:

.. code-block:: bash

    learn --method=without --model=RB
    learn --method=implicit --model=RB
    learn --method=soft --model=RB

View available options:

.. code-block:: bash

    learn --help
