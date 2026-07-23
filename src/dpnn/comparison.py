"""
Pure Python comparison script for Poisson structure learning.

Usage as Python:
    from dpnn.comparison import ComparisonConfig, ComparisonRunner
    
    config = ComparisonConfig(
        model="RB",
        steps=100,
        methods=["implicit", "soft", "without"],
        folder_name="TEST"
    )
    runner = ComparisonRunner(config)
    runner.run()

Usage as CLI:
    python comparison.py --model=RB --steps=100 --implicit --soft --without --folder_name=TEST
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
import sys
import argparse
from pathlib import Path
from math import sqrt
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from dpnn.simulation import simulate_batch, save_simulation
from dpnn.training import (
    Learner,
    LearnerIMR,
    LearnerRK4,
    check_folder,
    DEFAULT_folder_name,
    DEFAULT_dataset,
    DEFAULT_jacobi_loss_mode,
    DEFAULT_hutchinson_samples,
)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class ComparisonConfig:
    """Configuration for comparison run."""
    # Model and simulation
    model: str = "RB"
    steps: int = 100
    scheme: str = "IMR"
    
    # Methods to compare
    methods: List[str] = field(default_factory=lambda: ["implicit", "soft", "without"])
    
    # Initial conditions
    init_mx: float = 10.0
    init_my: float = 3.0
    init_mz: float = 4.0
    init_rx: float = 1.0
    init_ry: float = -3.0
    init_rz: float = 10.0
    
    # Model parameters
    Ix: float = 10.0
    Iy: float = 20.0
    Iz: float = 40.0
    Mgl: float = 9.81 * 0.1
    M_tau: float = 0.0
    M: float = 0.5
    
    # Learning parameters
    neurons: int = 64
    layers: int = 2
    batch_size: int = 32
    dt: float = 0.0  # 0.0 = automatic
    alpha: float = 2.0
    lr: float = 0.001
    epochs: int = 60
    prefactor: float = 1.0
    jac_prefactor: float = 1.0
    dropout_rate: float = 0.3
    quad_features: bool = False
    const_L: bool = False
    
    # Data
    generate: bool = False
    external_data: Optional[str] = None
    external_data_simple_format: bool = False
    sampling: int = 100
    points: int = 100
    seed: int = 42
    
    # Jacobi loss
    jacobi_loss_mode: str = DEFAULT_jacobi_loss_mode
    hutchinson_samples: int = DEFAULT_hutchinson_samples
    
    # Misc
    folder_name: str = DEFAULT_folder_name
    cuda: bool = False
    multiprocessing: bool = False
    simulation_batch_size: int = 256
    no_data_to_gpu: bool = True
    theta_sampling: int = 20
    dimensions: int = 10
    zeta: float = 0.0
    verbose: bool = False
    no_show: bool = False
    
    # Auto-computed
    device: torch.device = field(init=False)
    
    def __post_init__(self):
        """Set device after initialization."""
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            # When using CPU, always keep data on CPU (no_data_to_gpu=True means keep data on CPU)
            self.no_data_to_gpu = True
            if self.verbose:
                print("Using CPU")
    
    def to_namespace(self):
        """Convert to argparse.Namespace for backward compatibility with helper functions."""
        ns = argparse.Namespace(**self.__dict__)
        # Add boolean flags for methods
        ns.implicit = "implicit" in self.methods
        ns.soft = "soft" in self.methods
        ns.without = "without" in self.methods
        return ns


# ============================================================================
# COMPARISON RUNNER
# ============================================================================

class ComparisonRunner:
    """Runs comparison of different learning methods."""
    
    def __init__(self, config: ComparisonConfig):
        """Initialize with configuration."""
        self.config = config
        self._validate_config()
        self._setup()
    
    def _validate_config(self):
        """Validate configuration."""
        if "implicit" in self.config.methods and self.config.model in ["HT", "P3D", "K3D", "P2D", "Sh", "D"]:
            raise ValueError(f"Implicit solver not yet implemented for {self.config.model}.")
        
        if self.config.model == "K3D" and self.config.scheme != "IMR":
            raise ValueError("Don't use CN for Kepler.")
        
        if self.config.model in ("P2D", "Sh") and self.config.scheme != "IMR":
            raise ValueError(f"Only use the IMR scheme for {self.config.model}.")
        
        if (("implicit" in self.config.methods or "soft" in self.config.methods) and self.config.const_L):
            raise ValueError("Only use constant L with the without method.")
        
        if self.config.generate and self.config.external_data:
            raise ValueError("Cannot use both generate and external_data. Choose one.")
        
        if self.config.external_data_simple_format and not self.config.external_data:
            raise ValueError("Cannot use external_data_simple_format without external_data.")
    
    def _setup(self):
        """Setup folders and save configuration."""
        check_folder(self.config.folder_name)
        
        # Save config to file
        config_path = Path(self.config.folder_name) / "config.txt"
        with open(config_path, 'w') as f:
            for key, value in self.config.__dict__.items():
                if key != 'device':  # Skip device object
                    f.write(f"{key}: {value}\n")
        
        if self.config.verbose:
            print(f"Configuration saved to: {config_path}")
    
    def _get_learner(self, method: str) -> Learner:
        """Create and return appropriate learner."""
        learner_kwargs = {
            "model": self.config.model,
            "neurons": self.config.neurons,
            "layers": self.config.layers,
            "batch_size": self.config.batch_size,
            "dt": self.config.dt,
            "name": self.config.folder_name,
            "device": self.config.device,
            "dropout_rate": self.config.dropout_rate,
            "quad_features": self.config.quad_features,
            "simulation_batch_size": self.config.simulation_batch_size,
            "no_data_to_gpu": self.config.no_data_to_gpu,
            "jacobi_loss_mode": self.config.jacobi_loss_mode,
            "hutchinson_samples": self.config.hutchinson_samples,
            "external_data_path": self.config.external_data,
            "external_data_simple_format": self.config.external_data_simple_format,
            "verbose": self.config.verbose,
        }
        
        if self.config.model in ["HT", "P3D", "K3D", "P2D", "Sh", "D"]:
            learner_kwargs["D"] = self.config.dimensions
        
        if method in ["soft", "without"]:
            learner_kwargs["use_constant_L"] = self.config.const_L
        
        # Select learner class based on scheme
        if self.config.scheme == "IMR":
            return LearnerIMR(**learner_kwargs)
        elif self.config.scheme == "RK4":
            return LearnerRK4(**learner_kwargs)
        else:
            return Learner(**learner_kwargs)
    
    def run(self):
        """Run the full comparison."""
        import gc
        
        print("=" * 70)
        print("COMPARISON RUNNER - Pure Python")
        print("=" * 70)
        
        # Convert config to namespace for backward compatibility
        args = self.config.to_namespace()
        
        # Auto-compute dt if 0.0
        if args.dt <= 0.0:
            args.dt = resolve_automatic_dt(args)
            self.config.dt = args.dt
        
        # Generate or load trajectories
        print("\n" + "-" * 70)
        print("Preparing trajectories (generating, loading, or simulating GT)")
        print("-" * 70)
        start_time = time.time()
        generate_trajectories(args)
        end_time = time.time()
        print(f"Time: {end_time - start_time:.2f} seconds")
        
        # Clear memory after trajectory generation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train learners for each method
        if "implicit" in self.config.methods:
            print("\n" + "-" * 70)
            print("Training: Implicit Jacobi")
            print("-" * 70)
            learner = self._get_learner("implicit")
            learner.learn(
                method="implicit",
                learning_rate=self.config.lr,
                epochs=self.config.epochs,
                prefactor=self.config.prefactor
            )
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if "soft" in self.config.methods:
            print("\n" + "-" * 70)
            print("Training: Soft Jacobi")
            print("-" * 70)
            learner = self._get_learner("soft")
            learner.learn(
                method="soft",
                learning_rate=self.config.lr,
                epochs=self.config.epochs,
                prefactor=self.config.prefactor,
                jac_prefactor=self.config.jac_prefactor
            )
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if "without" in self.config.methods:
            print("\n" + "-" * 70)
            print("Training: Without Jacobi")
            print("-" * 70)
            learner = self._get_learner("without")
            learner.learn(
                method="without",
                learning_rate=self.config.lr,
                epochs=self.config.epochs,
                prefactor=self.config.prefactor
            )
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Plot results
        if not self.config.no_show:
            print("\n" + "-" * 70)
            print("Plotting training errors")
            print("-" * 70)
            plot_training_errors(args)
        
        print("\n" + "=" * 70)
        print("COMPARISON COMPLETE")
        print("=" * 70)


def norm(x, y, z):
    """
    The norm function calculates the magnitude of a vector in three-dimensional space.
    
    :param x: The parameter "x" represents the value of the x-coordinate in a three-dimensional space
    :param y: The parameter "y" represents the y-coordinate of a point in a three-dimensional space
    :param z: The parameter "z" represents the value of the z-coordinate in a three-dimensional space
    :return: the square root of the sum of the squares of the three input values (x, y, and z).
    """
    return sqrt(x**2+y**2+z**2)

def generate_initial_conditions(args, device="cpu"):
    
    def sample_within_ball(radius, batch_size, device):
        vec = torch.randn(batch_size, 3, device=device)
        vec /= vec.norm(dim=1, keepdim=True)
        # use cube root to ensure uniform distribution within the ball
        scale = torch.rand(batch_size, 1, device=device).pow(1/3)
        return vec * scale * radius
    
    m_radius = norm(args.init_mx, args.init_my, args.init_mz)
    r_radius = norm(args.init_rx, args.init_ry, args.init_rz)

    init_m = sample_within_ball(m_radius, args.points, device)
    init_r = sample_within_ball(r_radius, args.points, device)
    return torch.cat([init_m, init_r], dim=1)

def load_initial_conditions(filename, device="cpu"):
    """
    Loads initial conditions from a CSV file and returns a PyTorch tensor.
    """
    df = pd.read_csv(filename)
    
    # Extract the initial conditions columns and convert to a NumPy array
    init_m = df[['init_mx', 'init_my', 'init_mz']].values
    init_r = df[['init_rx', 'init_ry', 'init_rz']].values
    
    # Concatenate and convert to a PyTorch tensor
    initial_conditions = torch.tensor(
        np.concatenate([init_m, init_r], axis=1), 
        dtype=torch.float32, 
        device=device
    )
    
    print(f"Loaded {len(initial_conditions)} initial conditions from: {filename}")
    return initial_conditions


def split_into_batches(data, batch_size):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

def simulate_batch_normal(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate_batch(args, initial_conditions_batch, method="normal")

def simulate_batch_implicit(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate_batch(args, initial_conditions_batch, method="implicit")

def simulate_batch_soft(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate_batch(args, initial_conditions_batch, method="soft")

def simulate_batch_without(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate_batch(args, initial_conditions_batch, method="without")

def run_simulation(sim_func, argss, use_multiprocessing):
    if use_multiprocessing:
        ctx = mp.get_context('spawn')
        with ctx.Pool(3) as pool:
            dfs = pool.map(sim_func, argss)
    else:
        dfs = [sim_func(args) for args in argss]
    return dfs

def models_exist(folder_name, method):
    """Check if saved models exist for a given method."""
    model_path = Path(folder_name) / "saved_models" / f"{method}_jacobi_energy"
    return model_path.exists()


def generate_trajectories(args):
    """
    The function `generate_trajectories` generates and saves trajectories based on the given arguments, either using a deterministic approach or simulating with learned models, or loads external data.
    
    :param args: The `args` parameter is a dictionary or object that contains various arguments or parameters for the `generate_trajectories` function. These arguments control the behavior of the function and determine what kind of trajectories are generated or simulated
    """
    total_generalization_data_frame = None
    if args.external_data:
        print(f"Loading external dataset from: {args.external_data}")
        total_generalization_data_frame = pd.read_csv(args.external_data, dtype=np.float32)
        # Save the external data to the default dataset path so Learner can find it
        save_simulation(total_generalization_data_frame, args.folder_name+"/"+DEFAULT_dataset)
    elif args.generate:
        print("Generating dataset.")
        #Now we generage initial conditions (deterministic)
        np.random.seed(args.seed)
        initial_conditions = generate_initial_conditions(args, device=args.device)
        initial_condition_batches = split_into_batches(initial_conditions, args.simulation_batch_size)
        batched_inputs = [(args, batch) for batch in initial_condition_batches]

        if args.multiprocessing:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                dfs = pool.map(simulate_batch_normal, batched_inputs)
        else:
            dfs = [simulate_batch_normal(x) for x in batched_inputs]

        #save
        print("Saving dataset")
        total_data_frame = pd.concat(dfs, ignore_index=False)
        #save to file
        save_simulation(total_data_frame, args.folder_name+"/"+DEFAULT_dataset)
        print("Generated trajectories saved to: ", args.folder_name+"/"+DEFAULT_dataset)
        total_generalization_data_frame = total_data_frame # Use generated data as GT
    else: #simulating with the learned models, first generate GT if not external data
        multiprocessing = args.multiprocessing
        print("-------------------------------")
        print("Simulating with learned models.")    
        print("-------------------------------")

        #GT
        print("Generating GT for comparison.")

        np.random.seed(args.seed+100*args.sampling) #new seed, safely beyond the last used value
        initial_conditions = generate_initial_conditions(args, device=args.device)

        initial_condition_batches = split_into_batches(initial_conditions, args.simulation_batch_size)
 
        batched_inputs = [(args, batch) for batch in initial_condition_batches]

        if args.multiprocessing:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                dfs = pool.map(simulate_batch_normal, batched_inputs)
        else:
            dfs = [simulate_batch_normal(x) for x in batched_inputs]

        total_generalization_data_frame = pd.concat(dfs, ignore_index=True)
        save_simulation(total_generalization_data_frame, args.folder_name+"/data/generalization.xyz")

    # Now simulate with learned models for comparison if needed
    total_implicit_data_frame = None
    total_soft_data_frame = None
    total_without_data_frame = None

    # These simulations will use the initial conditions that were either generated or from the external data for GT
    # The initial_conditions for these simulations should come from the *first* states of total_generalization_data_frame
    if total_generalization_data_frame is not None:
        # Extract initial conditions from the loaded/generated GT data
        # This assumes the GT data has columns like 'old_mx', 'old_my', 'old_mz', 'old_rx', 'old_ry', 'old_rz'
        # which represent the initial state of each trajectory.
        # This part might need adjustment based on the exact format of the external data.
        if args.model == "RB":
            initial_m_cols = ["old_mx", "old_my", "old_mz"]
            initial_r_cols = [] # RB doesn't use r
        elif args.model in ["HT", "P3D", "K3D"]:
            initial_m_cols = ["old_mx", "old_my", "old_mz"]
            initial_r_cols = ["old_rx", "old_ry", "old_rz"]
        elif args.model == "P2D":
            initial_m_cols = ["old_mx", "old_my"]
            initial_r_cols = ["old_rx", "old_ry"]
        elif args.model == "Sh": # Assuming 'u', 'x', 'y', 'z' for Shivamoggi as per dataset.py
            initial_m_cols = ["old_u"] # u is like momentum here
            initial_r_cols = ["old_x", "old_y", "old_z"]
        elif args.model == "D":
            initial_r_cols = [f"old_r{i}" for i in range(args.dimensions)]
            initial_p_cols = [f"old_p{i}" for i in range(args.dimensions)]
            initial_m_cols = initial_p_cols
        else:
            raise Exception("Unknown model for extracting initial conditions from GT data.")

        # Get the unique initial conditions (the first row of each trajectory)
        # Assuming that the 'time' column starts from 0 for each unique trajectory.
        # We need to get the initial conditions for *all* trajectories, not just for a single batch.
        # A simpler approach might be to store the initial_conditions in the args object if it's generated,
        # or load them separately if they are from external data.
        # For now, let's assume `total_generalization_data_frame` has a 'time' column and we want rows where time is 0.
        initial_states_df = total_generalization_data_frame[total_generalization_data_frame['time'] == 0]

        if args.model == "RB":
            initial_conditions_for_learned = torch.tensor(initial_states_df[initial_m_cols].values, dtype=torch.float32, device=args.device)
            # Pad with zeros for r, if the simulate_batch expects 6 dimensions
            initial_conditions_for_learned = torch.cat([initial_conditions_for_learned, torch.zeros(initial_conditions_for_learned.shape[0], 3, device=args.device)], dim=1)
        elif args.model == "Sh":
            initial_conditions_for_learned = torch.tensor(initial_states_df[initial_m_cols + initial_r_cols].values, dtype=torch.float32, device=args.device)
        elif args.model in ["HT", "P3D", "K3D", "P2D"]:
            initial_conditions_for_learned = torch.tensor(initial_states_df[initial_m_cols + initial_r_cols].values, dtype=torch.float32, device=args.device)
        elif args.model == "D":
            initial_conditions_for_learned = torch.tensor(initial_states_df[initial_m_cols + initial_r_cols].values, dtype=torch.float32, device=args.device)

        initial_condition_batches_for_learned = split_into_batches(initial_conditions_for_learned, args.simulation_batch_size)
        batched_inputs_for_learned = [(args, batch) for batch in initial_condition_batches_for_learned]

        # Skip learned model simulation on --generate runs (models haven't been trained yet)
        if args.generate:
            print(">>> Skipping learned model simulation (--generate: models not trained yet)")
        else:
            if args.implicit and models_exist(args.folder_name, "implicit"):
                print(">>> Simulating with learned implicit.")
                print(f"    Loading models from: {args.folder_name}/saved_models/implicit_jacobi_*")
                if args.multiprocessing:
                    ctx = mp.get_context('spawn')
                    with ctx.Pool(processes=min(3, len(batched_inputs_for_learned))) as pool:
                        print(f"    Starting multiprocessing pool with {min(3, len(batched_inputs_for_learned))} processes")
                        dfs = pool.map(simulate_batch_implicit, batched_inputs_for_learned)
                else:
                    print(f"    Starting sequential simulation ({len(batched_inputs_for_learned)} batches)")
                    dfs = [simulate_batch_implicit(x) for x in batched_inputs_for_learned]
                print(f"    Concatenating {len(dfs)} dataframes...")
                total_implicit_data_frame = pd.concat(dfs, ignore_index=True)
                save_simulation(total_implicit_data_frame, args.folder_name+"/data/learned_implicit.xyz")
                print(">>> Learned implicit simulation complete")

            if args.soft and models_exist(args.folder_name, "soft"):
                print(">>> Simulating with learned soft.")
                print(f"    Loading models from: {args.folder_name}/saved_models/soft_jacobi_*")
                if args.multiprocessing:
                    ctx = mp.get_context('spawn')
                    with ctx.Pool(processes=min(3, len(batched_inputs_for_learned))) as pool:
                        print(f"    Starting multiprocessing pool with {min(3, len(batched_inputs_for_learned))} processes")
                        dfs = pool.map(simulate_batch_soft, batched_inputs_for_learned)
                else:
                    print(f"    Starting sequential simulation ({len(batched_inputs_for_learned)} batches)")
                    dfs = [simulate_batch_soft(x) for x in batched_inputs_for_learned]
                print(f"    Concatenating {len(dfs)} dataframes...")
                total_soft_data_frame = pd.concat(dfs, ignore_index=True)
                save_simulation(total_soft_data_frame, args.folder_name+"/data/learned_soft.xyz")
                print(">>> Learned soft simulation complete")

            if args.without and models_exist(args.folder_name, "without"):
                print(">>> Simulating with learned without.")
                print(f"    Loading models from: {args.folder_name}/saved_models/without_jacobi_*")
                if args.multiprocessing:
                    ctx = mp.get_context('spawn')
                    with ctx.Pool(processes=min(3, len(batched_inputs_for_learned))) as pool:
                        print(f"    Starting multiprocessing pool with {min(3, len(batched_inputs_for_learned))} processes")
                        dfs = pool.map(simulate_batch_without, batched_inputs_for_learned)
                else:
                    print(f"    Starting sequential simulation ({len(batched_inputs_for_learned)} batches)")
                    dfs = [simulate_batch_without(x) for x in batched_inputs_for_learned]
                print(f"    Concatenating {len(dfs)} dataframes...")
                total_without_data_frame = pd.concat(dfs, ignore_index=True)
                save_simulation(total_without_data_frame, args.folder_name+"/data/learned_without.xyz")
                print(">>> Learned without simulation complete")
    else:
        print("No generalization data available to simulate with learned models.")

    # Always save the generalization data (either generated or loaded external data)
    # The GT data is already saved to DEFAULT_dataset path earlier if generated or external.
    # We also save it to 'generalization.xyz' for consistency in output.
    if total_generalization_data_frame is not None:
        save_simulation(total_generalization_data_frame, args.folder_name+"/data/generalization.xyz")

def add_plot(ax, x=None,y=None, name=""):
    """
    The function `add_plot` adds a line plot to pyplot, with the option to specify x and y data and a label for the plot.
    
    :param ax: The matplotlib.pyplot module (or axes object)
    :param x: The x-axis values for the plot. If provided, the plot will be a line plot with x and y values. If not provided, the plot will be a line plot with only y values
    :param y: The `y` parameter is a list or array of values that represent the y-coordinates of the data points to be plotted
    :param name: The name parameter is a string that represents the label for the plot. It is used to identify the plot in the legend of the graph
    """
    if x is not None:
        ax.plot(x, y, lw=0.7, label=name)
    else:
        ax.plot(y, lw=0.7, label=name)    

def plot_training_errors(args):
    """
    The function `plot_training_errors` reads error data from CSV files and plots the training and validation errors for different scenarios.
    
    :param args: The `args` parameter is an object that contains the following attributes:\n    """
    print("***If Runtime tkinter errors are raised, it is because some matplotlib vs threads problems. Shouldn\'t be serious.***")

    name = args.folder_name
    if args.soft:
        df_soft_errors = pd.read_csv(name+"/data/errors_soft.csv")
        train_mov_errors = df_soft_errors["train_mov"]
        validation_mov_errors = df_soft_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="soft train move")
        add_plot(plt, y=validation_mov_errors[1:], name="soft val move")
        plt.legend()
        plt.show()

        validation_reg_errors = df_soft_errors["val_reg"]
        train_reg_errors = df_soft_errors["train_reg"]
        add_plot(plt, y=train_reg_errors[1:], name="soft train Jacobi")
        add_plot(plt, y=validation_reg_errors[1:], name="soft val Jacobi")
        plt.legend()
        plt.show()

    if args.implicit:
        df_implicit_errors = pd.read_csv(name+"/data/errors_implicit.csv")
        train_mov_errors = df_implicit_errors["train_mov"]
        validation_mov_errors = df_implicit_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="impclicit train move")
        add_plot(plt, y=validation_mov_errors[1:], name="implicit val move")
        plt.legend()
        plt.show()

    if args.without:
        df_without_errors = pd.read_csv(name+"/data/errors_without.csv")
        train_mov_errors = df_without_errors["train_mov"]
        validation_mov_errors = df_without_errors["val_mov"]
        add_plot(plt, y=train_mov_errors[1:], name="without train move")
        add_plot(plt, y=validation_mov_errors[1:], name="without val move")
        plt.legend()
        plt.show()

def resolve_automatic_dt(args):
    """
    The function `resolve_automatic_dt` calculates the time step `dt` based on the given model and
    parameters.
    
    :param args: args is a dictionary or object that contains the following parameters:
    :return: the value of dt, which is the time step for the simulation.
    """
    if args.model == "RB": #rigid body
        omega = sqrt(max([args.init_mx/args.Ix, args.init_mx/args.Ix, args.init_mz/args.Iz]))
    elif args.model == "HT": #heavy top
        omega1 = sqrt(max([args.init_mx/args.Ix, args.init_mx/args.Ix, args.init_mz/args.Iz]))
        omega2 = sqrt(args.Mgl*args.init_rz)*sqrt(max([1.0/args.Ix, 1.0/args.Iy, 1.0/args.Iz]))
        omega = max(omega1, omega2)
    elif args.model == "P3D": #3D Harmonic oscillator
        omega = sqrt(args.alpha/args.M) 
    elif args.model == "P2D": #3D Harmonic oscillator
        omega = sqrt(args.alpha/args.M) 
    elif args.model == "K3D": #3D Kepler problem
        r = math.sqrt(args.init_rx**2+args.init_ry**2+args.init_rz**2)
        p = math.sqrt(args.init_mx**2+args.init_my**2+args.init_mz**2)
        m = r*p
        e = args.M*args.alpha**2/(2*m**2)
        omega = 2/(args.alpha*math.sqrt(args.M/(2*e**3))) # increased for stability
    elif args.model == "Sh": #Shivamoggi
        omega = 2*math.pi*2 # increased for stability
    elif args.model == "D":
        omega = sqrt(args.alpha/args.M)
    else:
        raise Exception("Unkonown model.")
    dt = 0.01 * 2*math.pi/omega
    print("Setting dt = ", dt)
    return dt
        

# The above code is a Python script that performs a comparison between different numerical schemes for
# a given model. It takes command line arguments to specify the parameters of the simulation, such as
# the numerical scheme, model, number of simulation steps, initial momentum and position values,
# potential magnitude, and more.
def main():
    """
    Pure Python main entry point for comparison script.
    
    Supports CLI via argparse for backward compatibility, but uses
    ComparisonRunner internally for clean code.
    """
    parser = argparse.ArgumentParser(
        description="Compare different Poisson structure learning methods"
    )
    # Model and simulation
    parser.add_argument("--model", default="RB", type=str, help="Model: RB, HT, P3D, K3D, P2D or D")
    parser.add_argument("--steps", default=100, type=int, help="Number of simulation steps")
    parser.add_argument("--scheme", default="IMR", type=str, help="Integration scheme: IMR, RK4, etc.")
    
    # Methods to compare
    parser.add_argument("--implicit", action="store_true", help="Include implicit Jacobi method")
    parser.add_argument("--soft", action="store_true", help="Include soft Jacobi method")
    parser.add_argument("--without", action="store_true", help="Include without Jacobi method")
    
    # Initial conditions
    parser.add_argument("--init_mx", default=10.0, type=float, help="Initial momentum x")
    parser.add_argument("--init_my", default=3.0, type=float, help="Initial momentum y")
    parser.add_argument("--init_mz", default=4.0, type=float, help="Initial momentum z")
    parser.add_argument("--init_rx", default=1.0, type=float, help="Initial position x")
    parser.add_argument("--init_ry", default=-3.0, type=float, help="Initial position y")
    parser.add_argument("--init_rz", default=10.0, type=float, help="Initial position z")
    
    # Model parameters
    parser.add_argument("--Ix", default=10.0, type=float, help="Moment of inertia Ix")
    parser.add_argument("--Iy", default=20.0, type=float, help="Moment of inertia Iy")
    parser.add_argument("--Iz", default=40.0, type=float, help="Moment of inertia Iz")
    parser.add_argument("--Mgl", default=9.81*0.1, type=float, help="M*g*l parameter")
    parser.add_argument("--M_tau", default=0.0, type=float, help="Energy regularization parameter")
    parser.add_argument("--M", default=0.5, type=float, help="Mass")
    
    # Learning parameters
    parser.add_argument("--neurons", default=64, type=int, help="Number of neurons")
    parser.add_argument("--layers", default=2, type=int, help="Number of layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--dt", default=0.0, type=float, help="Timestep (0.0 = automatic)")
    parser.add_argument("--alpha", default=2.0, type=float, help="Potential magnitude/tau prefactor")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--epochs", default=60, type=int, help="Number of training epochs")
    parser.add_argument("--prefactor", default=1.0, type=float, help="Loss prefactor")
    parser.add_argument("--jac_prefactor", default=1.0, type=float, help="Jacobi loss prefactor")
    parser.add_argument("--dropout_rate", default=0.3, type=float, help="Dropout rate")
    parser.add_argument("--quad_features", action="store_true", help="Add quadratic features")
    parser.add_argument("--const_L", action="store_true", help="Use constant L matrix")
    
    # Data
    parser.add_argument("--generate", action="store_true", help="Generate new trajectories")
    parser.add_argument("--external_data", default=None, type=str, help="Path to external data CSV")
    parser.add_argument("--external_data_simple_format", action="store_true", help="Use simple format for external data")
    parser.add_argument("--sampling", default=100, type=int, help="Sampling points")
    parser.add_argument("--points", default=100, type=int, help="Points for generalization")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    
    # Jacobi loss
    parser.add_argument("--jacobi_loss_mode", default=DEFAULT_jacobi_loss_mode, type=str,
                        choices=["manual", "exact", "exact_backward", "hutchinson", "hutchinson_batch", "spectral"],
                        help="Jacobi loss evaluation mode")
    parser.add_argument("--hutchinson_samples", default=DEFAULT_hutchinson_samples, type=int,
                        help="Hutchinson probe vectors")
    
    # Misc
    parser.add_argument("--folder_name", default=DEFAULT_folder_name, type=str, help="Output folder")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--multiprocessing", action="store_true", help="Use multiprocessing")
    parser.add_argument("--simulation_batch_size", default=256, type=int, help="Simulation batch size")
    parser.add_argument("--no_data_to_gpu", action="store_true", help="Don't move data to GPU")
    parser.add_argument("--dimensions", default=10, type=int, help="Dimension for generic particle system")
    parser.add_argument("--zeta", default=0.0, type=float, help="Dissipation coefficient")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots")
    parser.add_argument("--theta_sampling", default=20, type=int, help="Theta sampling")
    
    args = parser.parse_args()
    
    # Collect methods
    methods = []
    if args.implicit:
        methods.append("implicit")
    if args.soft:
        methods.append("soft")
    if args.without:
        methods.append("without")
    
    # Default to all methods if none specified
    if not methods:
        methods = ["implicit", "soft", "without"]
    
    # Create config
    config = ComparisonConfig(
        model=args.model,
        steps=args.steps,
        scheme=args.scheme,
        methods=methods,
        init_mx=args.init_mx,
        init_my=args.init_my,
        init_mz=args.init_mz,
        init_rx=args.init_rx,
        init_ry=args.init_ry,
        init_rz=args.init_rz,
        Ix=args.Ix,
        Iy=args.Iy,
        Iz=args.Iz,
        Mgl=args.Mgl,
        M_tau=args.M_tau,
        M=args.M,
        neurons=args.neurons,
        layers=args.layers,
        batch_size=args.batch_size,
        dt=args.dt,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        prefactor=args.prefactor,
        jac_prefactor=args.jac_prefactor,
        dropout_rate=args.dropout_rate,
        quad_features=args.quad_features,
        const_L=args.const_L,
        generate=args.generate,
        external_data=args.external_data,
        external_data_simple_format=args.external_data_simple_format,
        sampling=args.sampling,
        points=args.points,
        seed=args.seed,
        jacobi_loss_mode=args.jacobi_loss_mode,
        hutchinson_samples=args.hutchinson_samples,
        folder_name=args.folder_name,
        cuda=args.cuda,
        multiprocessing=args.multiprocessing,
        simulation_batch_size=args.simulation_batch_size,
        no_data_to_gpu=not args.no_data_to_gpu,  # Note: invert the flag
        theta_sampling=args.theta_sampling,
        dimensions=args.dimensions,
        zeta=args.zeta,
        verbose=args.verbose,
        no_show=args.no_show,
    )
    
    # Run comparison
    runner = ComparisonRunner(config)
    runner.run()


if __name__ == "__main__":
    main()

