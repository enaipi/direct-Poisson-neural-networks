from re import I
import os, sys
import argparse
import simulate
import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from math import sqrt
from learn import *
import torch.multiprocessing as mp
import torch


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
    return simulate.simulate_batch(args, initial_conditions_batch, method="normal")

def simulate_batch_implicit(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate.simulate_batch(args, initial_conditions_batch, method="implicit")

def simulate_batch_soft(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate.simulate_batch(args, initial_conditions_batch, method="soft")

def simulate_batch_without(batch_initial_conditions_and_args):
    args, initial_conditions_batch = batch_initial_conditions_and_args
    return simulate.simulate_batch(args, initial_conditions_batch, method="without")

def run_simulation(sim_func, argss, use_multiprocessing):
    if use_multiprocessing:
        ctx = mp.get_context('spawn')
        with ctx.Pool(3) as pool:
            dfs = pool.map(sim_func, argss)
    else:
        dfs = [sim_func(args) for args in argss]
    return dfs

def generate_trajectories(args):
    """
    The function `generate_trajectories` generates and saves trajectories based on the given arguments, either using a deterministic approach or simulating with learned models.
    
    :param args: The `args` parameter is a dictionary or object that contains various arguments or parameters for the `generate_trajectories` function. These arguments control the behavior of the function and determine what kind of trajectories are generated or simulated
    """
    if args.generate:
        print("Generating dataset.")
        #Now we generage initial conditions (deterministic)
        np.random.seed(args.seed)

        initial_conditions = generate_initial_conditions(args, device=args.device)
        # initial_conditions = load_initial_conditions(args.folder_name+"/initial_conditions_generate.csv", device=args.device)
        
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
        simulate.save_simulation(total_data_frame, args.folder_name+"/"+DEFAULT_dataset)
        print("Generated trajectories saved to: ", args.folder_name+"/"+DEFAULT_dataset)

    else: #simulating with the learned models
        multiprocessing = args.multiprocessing

        total_implicit_data_frame = None
        total_soft_data_frame = None
        total_without_data_frame = None
        total_generalization_data_frame = None
        np.random.seed(args.seed+100*args.sampling) #new seed, safely beyond the last used value

        print("-------------------------------")
        print("Simulating with learned models.")    
        print("-------------------------------")

        #GT
        print("Generating GT.")

        initial_conditions = generate_initial_conditions(args, device=args.device)
        #initial_conditions = load_initial_conditions(args.folder_name+"/initial_conditions_test.csv", device=args.device)

        initial_condition_batches = split_into_batches(initial_conditions, args.simulation_batch_size)
 
        batched_inputs = [(args, batch) for batch in initial_condition_batches]

        if args.multiprocessing:
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                dfs = pool.map(simulate_batch_normal, batched_inputs)
        else:
            dfs = [simulate_batch_normal(x) for x in batched_inputs]

        total_generalization_data_frame = pd.concat(dfs, ignore_index=True)

        if args.implicit:
            print("Simulating with learned implicit.")
            if args.multiprocessing:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                    dfs = pool.map(simulate_batch_implicit, batched_inputs)
            else:
                dfs = [simulate_batch_implicit(x) for x in batched_inputs]
            total_implicit_data_frame = pd.concat(dfs, ignore_index=True)

        if args.soft:
            print("Simulating with learned soft.")
            if args.multiprocessing:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                    dfs = pool.map(simulate_batch_soft, batched_inputs)
            else:
                dfs = [simulate_batch_soft(x) for x in batched_inputs]
            total_soft_data_frame = pd.concat(dfs, ignore_index=True)

        if args.without:
            print("Simulating with learned without.")
            if args.multiprocessing:
                ctx = mp.get_context('spawn')
                with ctx.Pool(processes=min(3, len(batched_inputs))) as pool:
                    dfs = pool.map(simulate_batch_without, batched_inputs)
            else:
                dfs = [simulate_batch_without(x) for x in batched_inputs]
            total_without_data_frame = pd.concat(dfs, ignore_index=True)

        #    %if (args.points >= 10) and ((i % int(round(args.points/10))) == 0):
        #        print((i+0.0)/args.points*100, "%")

        #save to file
        if args.implicit:
            simulate.save_simulation(total_implicit_data_frame, args.folder_name+"/data/learned_implicit.xyz") 
        if args.soft:
            simulate.save_simulation(total_soft_data_frame, args.folder_name+"/data/learned_soft.xyz") 
        if args.without:
            simulate.save_simulation(total_without_data_frame, args.folder_name+"/data/learned_without.xyz") 
        simulate.save_simulation(total_generalization_data_frame, args.folder_name+"/data/generalization.xyz")
        

def add_plot(ax, x=None,y=None, name=""):
    """
    The function `add_plot` adds a line plot to a given matplotlib axes object, with the option to specify x and y data and a label for the plot.
    
    :param ax: The `ax` parameter is a matplotlib Axes object. It represents the subplot or axes on which the plot will be drawn
    :param x: The x-axis values for the plot. If provided, the plot will be a line plot with x and y values. If not provided, the plot will be a line plot with only y values
    :param y: The `y` parameter is a list or array of values that represent the y-coordinates of the data points to be plotted
    :param name: The name parameter is a string that represents the label for the plot. It is used to identify the plot in the legend of the graph
    """
    #ax.scatter(x[::args.plot_every],y[::args.plot_every])
    if x is not None:
        ax.plot(x, y, lw=0.7, label=name)
    else:
        ax.plot(y, lw=0.7, label=name)    

def plot_training_errors(args):
    """
    The function `plot_training_errors` reads error data from CSV files and plots the training and validation errors for different scenarios.
    
    :param args: The `args` parameter is an object that contains the following attributes:
    """
    print("***If Runtime tkinter errors are raised, it is because some matplotlib vs threads problems. Shouldn't be serious.***")

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
if __name__ == "__main__":
    #Parse arguments
    #Typical usage: python3 comparison.py --generate --steps=100 --implicit --soft --without
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", default="IMR", type=str, help="Numerical scheme. FE forward euler, BE backward euler, CN Crank-Nicholson, Eh Ehrenfest")
    parser.add_argument("--model", default="RB", type=str, help="Model: RB, HT, P3D, K3D, P2D or D.")
    parser.add_argument("--steps", default=100, type=int, help="Number of simulation steps")
    parser.add_argument("--init_mx", default=10.0, type=float, help="A value of momentum, x component")
    parser.add_argument("--init_my", default=3.0, type=float, help="A value of momentum, y component") 
    parser.add_argument("--init_mz", default=4.0, type=float, help="A value momentum, z component")
    parser.add_argument("--Mgl", default=9.81*0.1, type=float, help="M*g*l")
    parser.add_argument("--init_rx", default=1.0, type=float, help="Initial r, x component")
    parser.add_argument("--init_ry", default=-3.0, type=float, help="Initial r, y component")
    parser.add_argument("--init_rz", default=10.0, type=float, help="Initial r, z component")
    parser.add_argument("--Ix", default=10.0, type=float, help="Ix")
    parser.add_argument("--Iy", default=20.0, type=float, help="Iy")
    parser.add_argument("--Iz", default=40.0, type=float, help="Iz")
    parser.add_argument("--dt", default=0.0, type=float, help="Timestep, 0.0 for automatic")
    parser.add_argument("--alpha", default=2.0, type=float, help="Potential magnitude or tau prefactor.")
    parser.add_argument("--implicit", default=False, action="store_true", help="Use implicit Jacobi.")
    parser.add_argument("--soft", default=False, action="store_true", help="Use soft Jacobi.")
    parser.add_argument("--without", default=False, action="store_true", help="Use no Jacobi.")
    parser.add_argument("--normalise", default=False, action="store_true", help="Normalise energy matrix at the end")
    parser.add_argument("--generate", default=False, action="store_true", help="Generate new trajectories.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print a lot of useful output")
    parser.add_argument("--no_show", default=False, action="store_true", help="Don't show the training errors.")
    parser.add_argument("--sampling", default=100, type=int, help="Approximate number of points to be sampled on the sphere.")
    parser.add_argument("--points", default=100, type=int, help="Number of points on the sphere for generalization.")
    parser.add_argument("--prefactor", default=1.0, type=float, help="Loss prefactor")
    parser.add_argument("--jac_prefactor", default=1.0, type=float, help="Loss prefactor for Jacobi identity")
    parser.add_argument("--epochs", default=60, type=int, help="Number of epochs for soft.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--theta_sampling", default=20, type=int, help="Number theta angles.")
    parser.add_argument("--lr", default=0.001, type=float, help="Soft learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--neurons", default=64, type=int, help="Number of neurons.")
    parser.add_argument("--layers", default=2, type=int, help="Number of layers.")
    parser.add_argument("--M", default=0.5, type=float, help="mass")
    parser.add_argument("--folder_name", default=DEFAULT_folder_name, type=str, help="Folder name")
    parser.add_argument("--cuda", default=False, action="store_true", help="Use CUDA (under construction).")
    parser.add_argument("--zeta", default=0.0, type=float, help="Dissipation coefficient (NOT IMPLEMENTED)")
    parser.add_argument("--dropout_rate", default=0.3, type=float, help="Dropout rate")
    parser.add_argument("--quad_features", default=False, action="store_true", help="Adds quadratic features")
    parser.add_argument("--M_tau", default=0, type=float, help="Multiple of dt used for energy regularisation for training dataset and GT.")
    parser.add_argument("--multiprocessing", default=False, action="store_true", help="Use multiprocessing for simulation.")
    parser.add_argument("--simulation_batch_size", default=256, type=int, help="Batch size for simulation.")
    parser.add_argument("--no_data_to_gpu", default=True, action="store_false", help="Only move data to GPU while training.")
    parser.add_argument("--dimensions", default=10, type=int, help="The spatial dimension of the particle system 'D'.")
    parser.add_argument("--const_L", default=False, action="store_true", help="Whether to use a constant L matrix.")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    if args.dt == 0.0: #automatic
        args.dt = resolve_automatic_dt(args)

    check_folder(args.folder_name) #check whether the folders data and saved_models exist, or create them

    #save args to file
    original_stdout = sys.stdout
    with open(args.folder_name+'/args.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(args)
        sys.stdout = original_stdout

    if args.implicit and args.model in ["HT", "P3D", "K3D", "P2D", "Sh", "D"]:
        raise Exception(f"Implicit solver not yet implemented for {args.model}.")
    
    if args.model == "K3D" and args.scheme != "IMR":
        raise Exception("Don't use CN for Kepler.")

    if args.model in ("P2D", "Sh") and args.scheme != "IMR":
        raise Exception(f"Only use the IMR scheme for {args.model}.")
    
    if (args.implicit or args.soft) and args.const_L:
        raise Exception("Only use constant L with the without method.")

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        args.device = torch.device("cpu")
        args.no_data_to_gpu = False
        print("Using CPU")

    if args.generate:
        print("-------------------------------")
        print("Generating trajectories.")    
        print("-------------------------------")
        generate_trajectories(args)
        args.generate = False

    dissipative = False if (args.zeta == 0) else True
    if args.implicit:
        print("-------------------------------")
        print("Learning implicit Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
        elif args.scheme == "RK4":
            learner = LearnerRK4(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                            dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                            dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                            simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
            
        learner.learn(method = "implicit", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor)
    
    if args.soft:
        print("-------------------------------")
        print("Learning soft Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
        elif args.scheme == "RK4":
            learner = LearnerRK4(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                            dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                            dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                            simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu)
        learner.learn(method = "soft", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor, jac_prefactor = args.jac_prefactor)
    if args.without:
        print("-------------------------------")
        print("Learning without Jacobi.")    
        print("-------------------------------")
        if args.scheme == "IMR":
            learner = LearnerIMR(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu,
                                D=args.dimensions, use_constant_L=args.const_L)
        elif args.scheme == "RK4":
            learner = LearnerRK4(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                                dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                                dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                                simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu,
                                D=args.dimensions, use_constant_L=args.const_L)
        else:
            learner = Learner(model=args.model, neurons = args.neurons, layers = args.layers, batch_size = args.batch_size,
                            dt = args.dt, name = args.folder_name, device = args.device, dissipative = dissipative,
                            dropout_rate = args.dropout_rate, quad_features=args.quad_features,
                            simulation_batch_size=args.simulation_batch_size, no_data_to_gpu=args.no_data_to_gpu,
                            D=args.dimensions, use_constant_L=args.const_L)
        learner.learn(method = "without", learning_rate = args.lr, epochs = args.epochs, prefactor = args.prefactor)
    if not args.no_show:
        plot_training_errors(args)

    import time
    start_time = time.time()
    generate_trajectories(args)
    end_time = time.time()
    print(f"generate_trajectories runtime: {end_time - start_time:.2f} seconds")
    