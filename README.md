# Direct Poisson Neural Networks

[![Documentation](https://img.shields.io/badge/docs-latest-blue?style=flat-square)](https://enaipi.github.io/direct-Poisson-neural-networks/)

## Quick Start - Pure Python Scripts

All functionality is available through simple Python scripts (no installation required).

### 1. Command-Line Style (with argparse)

If you prefer command-line style, use Python scripts directly:

```bash
python comparison.py --generate --steps=1000 --implicit --soft --without --model=RB --folder_name=TEST --epochs=30 --lr=0.0001
python plot_compare.py --plot_RB_errors --GT --without --implicit --soft --folder_name=TEST
```

### 2. Clean Python API (Recommended)

For more control and clarity, use the Python API directly:

```python
from src.dpnn.comparison import ComparisonConfig, ComparisonRunner

# Configure the comparison
config = ComparisonConfig(
    model="RB",
    steps=1000,
    methods=["implicit", "soft", "without"],
    folder_name="my_test",
    neurons=64,
    layers=2,
    epochs=30,
    lr=0.0001,
    generate=True,  # Generate training data
)

# Run it
runner = ComparisonRunner(config)
runner.run()
```

### 3. See Examples

Check `example_comparison.py` for multiple examples:

```bash
python example_comparison.py
```

Examples included:
- Minimal configuration (all defaults)
- Rigid Body comparison
- Heavy Top soft method only
- Custom configuration with RK4 scheme

## Supported Models

All the following models work with the same scripts:

- `--model=RB` - Rigid Body (3D)
- `--model=HT` - Heavy Top (6D)
- `--model=P3D` - Particle 3D (6D canonical phase space)
- `--model=K3D` - Kinematics 3D (6D, alias for P3D)
- `--model=P2D` - Particle 2D (4D with friction)
- `--model=Sh` - Shivamoggi particle ND (4D)

Example with Heavy Top:

```bash
# CLI style
python comparison.py --generate --steps=1000 --implicit --soft --without --model=HT --folder_name=TEST_HT --epochs=30 --lr=0.0001

# Python API
from src.dpnn.comparison import ComparisonConfig, ComparisonRunner
config = ComparisonConfig(model="HT", steps=1000, epochs=30, lr=0.0001, folder_name="TEST_HT", generate=True)
runner = ComparisonRunner(config)
runner.run()
```

## It is also possible to compare just training and validation losses 

```bash
python compare-train-errors
```

Or from Python:

```python
from src.dpnn.postprocessing.plot_compare import plot_training_errors
plot_training_errors(config)
```

## Alternatively, you can do that step by step

First generate dataset for training with:

    python simulate.py --generate --steps=10000 --model=RB

for the rigid body (or `--model=HT` for heavy top, `--model=P3D` for particle in 3D, `--model=P2D` for particle in 2D, `--model=K3D` for kinematics, or `--model=Sh` for Shivamoggi particle)

Then we train networks with different methods:

    python learn.py --method=without --model=RB --epochs=30
    python learn.py --method=soft --model=RB --epochs=30
    python learn.py --method=implicit --model=RB --epochs=30

(Choose the method: `without`, `soft`, or `implicit`)

Then choose different initial conditions and see how well our network fits the evolution. If the initial condition is too different we will not get a good fit. If it is the same we will fit perfectly. 

    python simulate.py --steps=500 --generate
    python simulate.py --steps=500 --implicit
    python simulate.py --steps=500 --soft
    python simulate.py --steps=500 --without

## And we can plot and see:

    python plot_compare.py --plot_m --plot_E --plot_L

Check training error and errors while learning. 

## Building the Documentation

Install documentation dependencies:

    pip install -e ".[docs]"

Build HTML documentation:

    cd doc
    make html

The built documentation will be available in `doc/_build/html/`

## Python API Usage

You can use the unified learning framework directly in Python for maximum flexibility:

### Direct Learner API

```python
from src.dpnn.training import Learner

# Train a learner with legacy API
learner = Learner(
    model="RB",
    neurons=64,
    layers=3,
    batch_size=32,
    dt=0.1,
    name="my_test"
)

learner.learn(
    method="soft",
    learning_rate=1e-5,
    epochs=100,
    prefactor=1.0
)
```

### New General Purpose API

```python
from src.dpnn.training import RobustLearner
from src.dpnn.system_spec import SystemSpec

# Use modern SystemSpec approach
learner = RobustLearner(
    system_spec=SystemSpec.rigid_body(),
    neurons=64,
    layers=3,
)

learner.learn(
    method="implicit",
    jacobi_loss_mode="spectral",
    epochs=100
)
```

### Comparison Runner (Most Flexible)

```python
from src.dpnn.comparison import ComparisonConfig, ComparisonRunner

# Full control over all parameters
config = ComparisonConfig(
    model="RB",
    methods=["without", "soft", "implicit"],
    neurons=128,
    layers=2,
    epochs=50,
    lr=0.001,
    jacobi_loss_mode="spectral",
    quad_features=True,
    cuda=True,
    folder_name="my_experiment",
    verbose=True,
)

runner = ComparisonRunner(config)
runner.run()

# Access results from config
print(f"Results saved to: {config.folder_name}")
```

## Architecture

The code uses a unified `RobustLearner` class that:
- Supports all 6 physical systems (RB, HT, P3D, K3D, P2D, Sh)
- Implements all 7 Jacobi loss variants
- Supports 3 training methods (without, soft, implicit)
- Supports both IMR and RK4 integration schemes
- Eliminates code duplication from old dual implementation
- Maintains full backward compatibility

See [PHASE1_VERIFICATION_REPORT.md](PHASE1_VERIFICATION_REPORT.md) for architectural details.

## Creating Custom Bash Scripts

If you prefer command-line interface, create a simple bash script:

```bash
#!/bin/bash
cd /path/to/direct-Poisson-neural-networks
python "$@"
```

Then use as:

    bash learn.sh learn.py --method=soft --model=RB

Or create aliases in your `.bashrc`:

```bash
alias comparison='python /path/to/comparison.py'
alias simulate='python /path/to/simulate.py'
alias learn='python /path/to/learn.py'
alias plot-compare='python /path/to/plot_compare.py'
```

## Typical arguments used for the training can be found in folder 

    typical_args

Please cite as [M. Šípka, M. Pavelka, O. Esen, and M. Grmela, Direct Poisson neural networks: learning non-symplectic mechanical systems, Journal of Physics A: Mathematical and Theoretical 56(49), 2023.](https://iopscience.iop.org/article/10.1088/1751-8121/ad0803)
