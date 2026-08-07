"""
EXAMPLE 4: Fermi-Pasta-Ulam Chain

Generate trajectories for a 1D Fermi-Pasta-Ulam (FPU) chain and learn
Hamiltonian dynamics from the generated time series.

System: N-particle canonical Hamiltonian chain
  State z = (q_0, ..., q_{N-1}, p_0, ..., p_{N-1})
  Fixed boundaries: q_{-1} = q_N = 0
  Energy H = kinetic + quadratic/cubic/quartic nearest-neighbor potential
  Structure L = canonical symplectic Poisson matrix

Run this file as:
    python examples/04_fpu.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpnn.models.physical_models import FPUIMR
from dpnn.system_spec import SystemSpec
from dpnn.training.hamiltonian_learner import HamiltonianLearner

from dpnn.utils._common import TransitionDataset, save_standard_json, save_transition_csv


STATE_DIMENSIONS = 8
NUM_TRAJECTORIES = 12
STEPS_PER_TRAJECTORY = 200
DT = 0.01
# All example outputs (data/ and saved_models/) are collected under this
# top-level results folder, which is created automatically if missing.
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "examples_fpu"
UNITS = {"position": "dimensionless", "momentum": "dimensionless", "time": "s"}


class TransitionDataset(Dataset):
    """Dataset of consecutive states z_n -> z_{n+1} for residual training."""

    def __init__(self, trajectories):
        features = []
        targets = []
        for trajectory in trajectories:
            features.append(trajectory[:-1])
            targets.append(trajectory[1:])

        features = np.vstack(features).astype(np.float32)
        targets = np.vstack(targets).astype(np.float32)

        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)
        self.mid = 0.5 * (self.features + self.targets)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.mid[idx]


def make_initial_conditions(num_trajectories, dimensions, seed=42):
    """Create small-amplitude initial FPU states."""
    rng = np.random.default_rng(seed)
    grid = np.arange(1, dimensions + 1, dtype=np.float32)
    mode = np.sin(np.pi * grid / (dimensions + 1))

    initial_conditions = []
    for _ in range(num_trajectories):
        amplitude = rng.uniform(0.08, 0.18)
        q0 = amplitude * mode
        q0 += rng.normal(0.0, 0.01, size=dimensions).astype(np.float32)
        p0 = rng.normal(0.0, 0.02, size=dimensions).astype(np.float32)
        initial_conditions.append((q0.astype(np.float32), p0.astype(np.float32)))

    return initial_conditions


def simulate_fpu_trajectories():
    """Generate FPU trajectories using the physical FPUIMR model."""
    print("=" * 70)
    print("EXAMPLE 4: FERMI-PASTA-ULAM CHAIN")
    print("=" * 70)

    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    energies = []
    initial_conditions = make_initial_conditions(NUM_TRAJECTORIES, STATE_DIMENSIONS)

    for q0, p0 in initial_conditions:
        model = FPUIMR(
            M=1.0,
            dt=DT,
            alpha=0.25,
            beta=1.0,
            k=1.0,
            init_q=q0,
            init_p=p0,
        )

        states = []
        energy_values = []
        for _ in range(STEPS_PER_TRAJECTORY):
            z = torch.cat([model.q, model.p], dim=1)
            states.append(z[0].detach().cpu().numpy().copy())
            energy_values.append(float(model.get_E(z)[0].detach().cpu()))
            model.m_new()

        trajectories.append(np.asarray(states, dtype=np.float32))
        energies.append(np.asarray(energy_values, dtype=np.float32))

    save_standard_json(trajectories, data_dir, filename="fpu_data.json", system_name="FermiPastaUlam", 
                       dt=DT, dimension=2 * STATE_DIMENSIONS, units=UNITS)
    save_transition_csv(trajectories, data_dir, filename="fpu_transitions.csv", 
                        dt=DT, component_dim=STATE_DIMENSIONS)

    energy_array = np.vstack(energies)
    relative_energy_drift = np.max(
        np.abs(energy_array - energy_array[:, :1]) / np.maximum(np.abs(energy_array[:, :1]), 1e-12)
    )

    print("Generated FPU trajectories:")
    print(f"  Shape: {np.asarray(trajectories).shape}")
    print(f"  Chain particles: {STATE_DIMENSIONS}")
    print(f"  State dimension: {2 * STATE_DIMENSIONS}")
    print(f"  Time step: {DT}")
    print(f"  Max relative energy drift: {relative_energy_drift:.3e}")
    print(f"  Standard JSON: {data_dir / 'fpu_data.json'}")
    print(f"  Transition CSV: {data_dir / 'fpu_transitions.csv'}")

    return trajectories


def train_fpu_learner(trajectories, epochs=5):
    """Train a generic HamiltonianLearner on FPU transition pairs."""
    system_spec = SystemSpec.custom(
        name="FermiPastaUlam",
        dimension=2 * STATE_DIMENSIONS,
        structure_tensor="symplectic",
        poisson_bracket_type="canonical",
        conserved_quantities=["energy"],
        description="FPU chain in canonical q,p coordinates",
        units={"position": "dimensionless", "momentum": "dimensionless", "time": "s"},
    )

    dataset = TransitionDataset(trajectories)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    learner = HamiltonianLearner(
        system_spec=system_spec,
        batch_size=32,
        neurons=64,
        layers=2,
        dt=DT,
        device=torch.device("cpu"),
        dropout_rate=0.1,
        jacobi_loss_mode="spectral",
        integration_scheme="imr",
        use_constant_L=False,
        name=str(OUTPUT_DIR),
        verbose=False,
    )
    learner.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    learner.valid_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    print("\nTraining FPU learner:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Epochs: {epochs}")
    print("  Method: soft")

    learner.learn(
        method="soft",
        learning_rate=1e-3,
        epochs=epochs,
        prefactor=1.0,
        jac_prefactor=1.0,
    )

    return learner


def inspect_learned_model(learner):
    """Print a small sanity check for learned H(z), L(z), and z_dot."""
    z = torch.randn(1, 2 * STATE_DIMENSIONS)
    H = learner.energy(z)
    L = learner.forward_L_tensor(z)
    z_dot = learner.compute_z_dot(z)

    print("\nLearned function shapes:")
    print(f"  H(z): {tuple(H.shape)}")
    print(f"  L(z): {tuple(L.shape)}")
    print(f"  z_dot: {tuple(z_dot.shape)}")
    print(f"  L antisymmetric: {torch.allclose(L + L.transpose(1, 2), torch.zeros_like(L))}")


def main():
    trajectories = simulate_fpu_trajectories()
    learner = train_fpu_learner(trajectories)
    inspect_learned_model(learner)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
