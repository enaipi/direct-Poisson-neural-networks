"""
EXAMPLE 5: Harmonic Oscillator Particles
 
Generate trajectories for N particles in D dimensions, each with an
independent harmonic potential, and learn the Hamiltonian dynamics
from the generated time series.
 
System: N particles in D dimensions, canonical Hamiltonian, no coupling
  State z = (q_0, ..., q_{N*D-1}, p_0, ..., p_{N*D-1})
  Energy H = sum_i p_i^2 / (2M) + 1/2 k sum_i q_i^2
  Structure L = canonical symplectic Poisson matrix
 
Run this file as:
    python examples/05_harmonic.py
"""

import sys
from pathlib import Path
 
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
 
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
 
from dpnn.models.physical_models import HarmonicIMR
from dpnn.system_spec import SystemSpec
from dpnn.training.hamiltonian_learner import HamiltonianLearner

from dpnn.utils._common import TransitionDataset, save_transition_csv, save_standard_json


N_PARTICLES = 4
DIMENSIONS = 2
STATE_DIMENSIONS = N_PARTICLES * DIMENSIONS 
NUM_TRAJECTORIES = 12
STEPS_PER_TRAJECTORY = 200
DT = 0.01
M = 1.0
K = 1.0
OUTPUT_DIR = Path("examples_harmonic")
UNITS = {"position": "dimensionless", "momentum": "dimensionless", "time": "s"}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_initial_conditions(num_trajectories, dimensions, seed=42):
    rng = np.random.default_rng(seed)
 
    initial_conditions = []
    for _ in range(num_trajectories):
        amplitude = rng.uniform(0.3, 0.8, size=dimensions).astype(np.float32)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=dimensions).astype(np.float32)
        q0 = amplitude * np.cos(phase)
        p0 = -amplitude * np.sin(phase)
        initial_conditions.append((q0.astype(np.float32), p0.astype(np.float32)))
 
    return initial_conditions


def simulate_harmonic_trajectories():
    """Generate harmonic oscillator trajectories using the physical HarmonicIMR model."""
    print("=" * 70)
    print("EXAMPLE 5: HARMONIC OSCILLATOR PARTICLES")
    print("=" * 70)
 
    data_dir = OUTPUT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
 
    trajectories = []
    energies = []
    initial_conditions = make_initial_conditions(NUM_TRAJECTORIES, STATE_DIMENSIONS)
 
    for q0, p0 in initial_conditions:
        model = HarmonicIMR(
            N=N_PARTICLES,
            D=DIMENSIONS,
            M=M,
            dt=DT,
            k=K,
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
 
    # labels "particle_idx"_"dimension"
    component_labels = [f"{n}_{d}" for n in range(N_PARTICLES) for d in range(DIMENSIONS)]
 
    save_standard_json(trajectories, data_dir, filename="harmonic_data.json",
        system_name="HarmonicOscillatorParticles", dt=DT,
        dimension=2 * STATE_DIMENSIONS, units=UNITS,)
    save_transition_csv(trajectories, data_dir, filename="harmonic_transitions.csv",
        dt=DT, component_dim=STATE_DIMENSIONS, component_labels=component_labels,)
 
    energy_array = np.vstack(energies)
    relative_energy_drift = np.max(
        np.abs(energy_array - energy_array[:, :1]) / np.maximum(np.abs(energy_array[:, :1]), 1e-12)
    )
 
    print("Generated harmonic oscillator trajectories:")
    print(f"  Shape: {np.asarray(trajectories).shape}")
    print(f"  Particles: {N_PARTICLES}, Dimensions: {DIMENSIONS}")
    print(f"  State dimension: {2 * STATE_DIMENSIONS}")
    print(f"  Time step: {DT}")
    print(f"  Max relative energy drift: {relative_energy_drift:.3e}")
    print(f"  Standard JSON: {data_dir / 'harmonic_data.json'}")
    print(f"  Transition CSV: {data_dir / 'harmonic_transitions.csv'}")
 
    return trajectories


def train_harmonic_learner(trajectories, epochs=5):
    """Train a generic HamiltonianLearner on harmonic oscillator transition pairs."""
    system_spec = SystemSpec.custom(
        name="HarmonicOscillatorParticles",
        dimension=2 * STATE_DIMENSIONS,
        structure_tensor="symplectic",
        poisson_bracket_type="canonical",
        conserved_quantities=["energy"],
        description="N particles in D dimensions, independent harmonic potential, canonical q,p coordinates",
        units=UNITS,
    )
 
    dataset = TransitionDataset(trajectories, DEVICE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
 
    learner = HamiltonianLearner(
        system_spec=system_spec,
        batch_size=32,
        neurons=64,
        layers=2,
        dt=DT,
        device=DEVICE,
        dropout_rate=0.1,
        jacobi_loss_mode="spectral",
        integration_scheme="imr",
        use_constant_L=False,
        verbose=False,
    )
    learner.train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    learner.valid_loader = DataLoader(val_data, batch_size=32, shuffle=False)
 
    print("\nTraining harmonic learner:")
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
    z = torch.randn(1, 2 * STATE_DIMENSIONS, device=DEVICE)
    H = learner.energy(z)
    L = learner.forward_L_tensor(z)
    z_dot = learner.compute_z_dot(z)
 
    print("\nLearned function shapes:")
    print(f"  H(z): {tuple(H.shape)}")
    print(f"  L(z): {tuple(L.shape)}")
    print(f"  z_dot: {tuple(z_dot.shape)}")
    print(f"  L antisymmetric: {torch.allclose(L + L.transpose(1, 2), torch.zeros_like(L))}")


def main():
    trajectories = simulate_harmonic_trajectories()
    learner = train_harmonic_learner(trajectories)
    inspect_learned_model(learner)
 
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}/")
 
 
if __name__ == "__main__":
    main()