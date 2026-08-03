"""Comparison and visualization of learned Poisson structures.

This module provides tools to:
- Load and process trajectory data
- Visualize learned vs ground truth Poisson structures
- Compute compatibility errors
- Compare old Learner vs GeneralSystemLearner models
"""

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os
import torch
from matplotlib import cm
from pathlib import Path

from dpnn.training import DEFAULT_folder_name
from dpnn import comparison

# Import from refactored modules
from .data_utils import (
    load_dataframes, filter_data, split_data_to_forward_paths,
    split_to_forward_paths, reject_outliers, normalize, load_normalized_Ls
)
from .visualization import (
    add_plot, plot_field, plot_errors_line, plot_errors_scatter, add_log
)
from .error_computation import (
    compat_error3D, compat_error_superintegrable, compat_error_6d_separable
)
from .model_generation import (
    load_learned_models, load_general_learner, get_poisson_structure,
    generate_E_points, generate_L_points
)


# Global variables
args = None
methods = {}
frames = []
titles = []


def get_frames_and_titles():
    """Get list of dataframes and titles to plot based on args."""
    fields = []
    titles_list = []
    
    for method_key in ['without', 'soft', 'implicit', 'GT', 'dataset']:
        if method_key in methods and methods[method_key]['df'] is not None:
            if (args.without and method_key == 'without') or \
               (args.soft and method_key == 'soft') or \
               (args.implicit and method_key == 'implicit') or \
               (args.GT and method_key == 'GT') or \
               (args.dataset and method_key == 'dataset'):
                fields.append(methods[method_key]["df"])
                titles_list.append(methods[method_key]["title"])
    
    return fields, titles_list


def plot_fields_from_dataframes(field_list):
    """Plot specified fields from dataframes."""
    file_name = ""
    for data_frame, name in zip(frames, titles):
        for col in field_list:
            if col not in data_frame.columns:
                continue
            plt.xlabel("time")
            add_plot(plt, data_frame["time"], data_frame[col], name=name+": "+col)
            file_name += "_"+name+"-"+col
    
    plt.legend()
    if args.export:
        file_name = args.folder_name+"/"+file_name+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name)
    plt.show()


def plot_first(field):
    """Plot only first trajectory for a field."""
    if not methods.get("GT"):
        print("Ground truth data required for plot_first")
        return
    
    dfgt = methods["GT"]["df"]
    if field not in dfgt.columns:
        print(f"Field {field} not found in data")
        return
    
    paths = split_to_forward_paths(dfgt, field)
    if len(paths) > 1:
        x_gt = paths[1]
        times = dfgt["time"][:len(x_gt)]
    else:
        x_gt = paths[0]
        times = dfgt["time"][:len(x_gt)]
    
    plt.figure()
    
    if args.soft and methods.get("soft"):
        dfls = methods["soft"]["df"]
        if field in dfls.columns:
            add_plot(plt, times, dfls[field][:len(times)], 
                    name=args.model + " soft: " + field)
    
    if args.implicit and methods.get("implicit"):
        dfli = methods["implicit"]["df"]
        if field in dfli.columns:
            add_plot(plt, times, dfli[field][:len(times)],
                    name=args.model + " implicit: " + field)
    
    if args.without and methods.get("without"):
        dflw = methods["without"]["df"]
        if field in dflw.columns:
            add_plot(plt, times, dflw[field][:len(times)],
                    name=args.model + " without: " + field)
    
    if args.GT:
        add_plot(plt, times, x_gt, name=args.model + " GT: " + field)
    
    plt.legend()
    if args.export:
        file_name = args.folder_name+"/"+args.model+"_first_"+field+".png"
        print("Exporting figure to: "+file_name)
        plt.savefig(file_name)
    plt.show()


def plot_fields_errors(fields, field_name=""):
    """Plot histograms of field errors (learned vs ground truth)."""
    if not methods.get("GT"):
        print("Warning: Ground truth data required for error plotting")
        return
    
    dfgt = methods["GT"]["df"]
    print(f"Plotting errors of fields: {fields}")
    file_name = args.model + "_" + field_name + "-errors"
    
    for method_key in ['without', 'soft', 'implicit']:
        if method_key not in methods or methods[method_key]['df'] is None:
            continue
        
        if (method_key == 'without' and not args.without) or \
           (method_key == 'soft' and not args.soft) or \
           (method_key == 'implicit' and not args.implicit):
            continue
        
        data_frame = methods[method_key]['df']
        name = methods[method_key]['title']
        
        values = {}
        gt = {}
        for field in fields:
            if field not in data_frame.columns or field not in dfgt.columns:
                print(f"Warning: Field {field} not found in data")
                continue
            values[field] = data_frame[field].values
            gt[field] = dfgt[field].values
        
        if not values:
            continue
        
        # Ensure both arrays have the same length (use minimum)
        min_len = min(len(v) for v in values.values()) if values else 0
        min_len = min(min_len, len(gt[list(gt.keys())[0]]) if gt else min_len)
        
        for field in list(values.keys()):
            values[field] = values[field][:min_len]
            gt[field] = gt[field][:min_len]
        
        # Compute total squared error for all fields
        total_error = np.sum(np.array([values[field] - gt[field] for field in fields if field in values]) ** 2, axis=0)
        
        if len(total_error) == 0:
            continue
        
        average_error = np.median(total_error)
        print(f"Median {field_name} error for {name}: {average_error}")
        
        file_name_current = file_name + "-" + name
        if args.export:
            add_log(file_name_current + " " + field_name, average_error)
        
        try:
            plt.figure()
            plt.hist(np.log10(total_error + 1e-10), bins=100, label=name, alpha=0.7)
            plt.legend()
            plt.xlabel(f"Trajectory errors (log10): {field_name}")
            plt.ylabel("Frequency")
            if args.export:
                file_name_out = args.folder_name + "/" + file_name_current + ".png"
                print(f"Exporting figure to: {file_name_out}")
                plt.savefig(file_name_out)
            plt.show()
        except Exception as e:
            print(f"Error plotting histogram: {e}")


def plot_compatibility_errors():
    """Plot L compatibility errors."""
    file_name = "compatibility_errors"
    
    for data_frame, name in zip(frames, titles):
        if args.model == "RB":
            file_name += "_RB"
            
            if "L_01" not in data_frame.columns:
                print(f"Warning: L matrix data not found for {name}")
                continue
            
            Ls = load_normalized_Ls(data_frame, 3)
            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]
            
            errors = compat_error3D(Ls, mxs, mys, mzs)
            iterations = np.linspace(1, len(errors), len(errors))
            
            average_error = np.trapz(errors) / len(errors)
            print(f"Average compatibility error for {name}: {average_error}")
            
            if args.export:
                add_log(file_name + " " + name, average_error)
            
            add_plot(plt, iterations, errors, name=name + ": compatibility_error")
            file_name += "-" + name
    
    plt.legend()
    if args.export:
        file_name_out = args.folder_name + "/" + file_name + ".png"
        print("Exporting figure to: " + file_name_out)
        plt.savefig(file_name_out)
    plt.show()


def plot_L_errors():
    """Plot L matrix reconstruction errors."""
    file_name = "L_errors"
    
    for data_frame, name in zip(frames, titles):
        if args.model == "RB":
            file_name += "_RB"
            
            Ls = load_normalized_Ls(data_frame, 3)
            mxs = data_frame["mx"]
            mys = data_frame["my"]
            mzs = data_frame["mz"]
            
            # Exact L for RigidBody
            Ls_exact = np.array([
                [[0.0, -mzs[i], mys[i]],
                 [mzs[i], 0.0, -mxs[i]],
                 [-mys[i], mxs[i], 0.0]]
                for i in range(len(mxs))
            ])
            Ls_exact = normalize(Ls_exact)
            
            total_error = [np.linalg.norm(Ls[i] - Ls_exact[i]) for i in range(len(Ls))]
            iterations = np.linspace(1, len(total_error), len(total_error))
            
            average_error = np.trapz(total_error) / len(total_error)
            print(f"Average L error for {name}: {average_error}")
            
            if args.export:
                add_log(file_name + " " + name, average_error)
            
            add_plot(plt, iterations, total_error, name=name + ": L_error")
            file_name += "-" + name
    
    plt.legend()
    if args.export:
        file_name_out = args.folder_name + "/" + file_name + ".png"
        print("Exporting figure to: " + file_name_out)
        plt.savefig(file_name_out)
    plt.show()


def plot_l_models():
    """Plot learned L models from neural networks."""
    models = load_learned_models(args.folder_name, args.model)
    
    for model_name, model_info in models.items():
        if args.model == "RB" and model_info.get('L'):
            L_tensor = model_info['L']
            mx, my, mz, L = generate_L_points(args, L_tensor)
            
            plt.figure()
            plt.title(model_name)
            plt.scatter(mx.detach().reshape(-1,), L.detach()[:,1,2], label="L23 vs mx")
            plt.scatter(my.detach().reshape(-1,), L.detach()[:,0,2], label="L13 vs my")
            plt.scatter(mz.detach().reshape(-1,), L.detach()[:,0,1], label="L12 vs mz")
            plt.legend()
            plt.show()


def plot_energy_models():
    """Plot learned energy models from neural networks."""
    models = load_learned_models(args.folder_name, args.model)
    
    for model_name, model_info in models.items():
        if model_info.get('energy'):
            energy = model_info['energy']
            
            if args.model == "RB":
                mx, my, mz, E = generate_E_points(args, energy)
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(mx.numpy(), my.numpy(), E.numpy(),
                                      cmap=cm.coolwarm, linewidth=0, antialiased=False)
                ax.set_xlabel("mx")
                ax.set_ylabel("my")
                ax.set_zlabel("Energy")
                ax.set_title(model_name)
                plt.show()


def main():
    """Main entry point for plot_compare script."""
    global args, methods, frames, titles
    
    parser = argparse.ArgumentParser(description="Compare learned vs ground truth Poisson structures")
    
    # Data selection
    parser.add_argument("--folder_name", default=DEFAULT_folder_name, type=str,
                       help="Folder with data and models")
    parser.add_argument("--plot_steps", default=None, type=int, help="Max steps to plot")
    parser.add_argument("--plot_every", default=1, type=int, help="Plot every nth point")
    
    # Data sources
    parser.add_argument("--without", default=False, action="store_true", help="Load Learned Without")
    parser.add_argument("--soft", default=False, action="store_true", help="Load Learned Soft")
    parser.add_argument("--implicit", default=False, action="store_true", help="Load Learned Implicit")
    parser.add_argument("--GT", default=False, action="store_true", help="Load Ground Truth")
    parser.add_argument("--dataset", default=False, action="store_true", help="Load Dataset")
    parser.add_argument("--general_learner", default="", type=str, help="Path to GeneralSystemLearner checkpoint")
    
    # Model and system
    parser.add_argument("--model", default="RB", type=str, help="Model: RB, HT, P2D, P3D, K3D, Sh")
    
    # Plotting options
    parser.add_argument("--plot_m", action="store_true", help="Plot momentum components")
    parser.add_argument("--plot_r", action="store_true", help="Plot position components")
    parser.add_argument("--plot_msq", action="store_true", help="Plot m squared")
    parser.add_argument("--plot_rsq", action="store_true", help="Plot r squared")
    parser.add_argument("--plot_field", default="None", type=str, help="Plot specific field")
    parser.add_argument("--plot_first", default="None", type=str, help="Plot first trajectory of field")
    parser.add_argument("--plot_first_mx", action="store_true", help="Plot first mx")
    parser.add_argument("--plot_first_rx", action="store_true", help="Plot first rx")
    
    # Error plotting
    parser.add_argument("--plot_compatibility", action="store_true", help="Plot L compatibility errors")
    parser.add_argument("--plot_L_errors", action="store_true", help="Plot L reconstruction errors")
    parser.add_argument("--plot_RB_errors", action="store_true", help="Plot RigidBody errors")
    parser.add_argument("--plot_HT_errors", action="store_true", help="Plot HeavyTop errors")
    parser.add_argument("--plot_training_errors", action="store_true", help="Plot training errors")
    
    # Model visualization
    parser.add_argument("--plot_Es", action="store_true", help="Plot energy from models")
    parser.add_argument("--plot_Ls", action="store_true", help="Plot L matrices from models")
    
    # Utility options
    parser.add_argument("--export", action="store_true", help="Save figures and logs")
    parser.add_argument("--no_clean", action="store_true", help="Don't clean log file")
    parser.add_argument("--init_mx", default=10.0, type=float, help="Initial mx for sampling")
    parser.add_argument("--init_my", default=3.0, type=float, help="Initial my for sampling")
    parser.add_argument("--init_mz", default=4.0, type=float, help="Initial mz for sampling")
    parser.add_argument("--init_rx", default=1.0, type=float, help="Initial rx for sampling")
    parser.add_argument("--init_ry", default=1.0, type=float, help="Initial ry for sampling")
    parser.add_argument("--init_rz", default=1.0, type=float, help="Initial rz for sampling")
    parser.add_argument("--density", default=60, type=int, help="Density for sampling grid")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.folder_name}...")
    
    data_sources = {}
    
    if args.dataset or args.without or args.soft or args.implicit or args.GT:
        try:
            dataset_path = Path(args.folder_name) / "data" / "dataset.xyz"
            if dataset_path.exists():
                data_sources['dataset'] = {
                    'df': pd.read_csv(dataset_path, nrows=args.plot_steps),
                    'title': 'Dataset'
                }
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
    
    if args.GT:
        try:
            gt_path = Path(args.folder_name) / "data" / "generalization.xyz"
            if gt_path.exists():
                data_sources['GT'] = {
                    'df': pd.read_csv(gt_path, nrows=args.plot_steps),
                    'title': 'Ground Truth'
                }
        except Exception as e:
            print(f"Warning: Could not load ground truth: {e}")
    
    if args.implicit:
        try:
            impl_path = Path(args.folder_name) / "data" / "learned_implicit.xyz"
            if impl_path.exists():
                data_sources['implicit'] = {
                    'df': pd.read_csv(impl_path, nrows=args.plot_steps),
                    'title': 'Learned Implicit'
                }
        except Exception as e:
            print(f"Warning: Could not load implicit learner data: {e}")
    
    if args.soft:
        try:
            soft_path = Path(args.folder_name) / "data" / "learned_soft.xyz"
            if soft_path.exists():
                data_sources['soft'] = {
                    'df': pd.read_csv(soft_path, nrows=args.plot_steps),
                    'title': 'Learned Soft'
                }
        except Exception as e:
            print(f"Warning: Could not load soft learner data: {e}")
    
    if args.without:
        try:
            without_path = Path(args.folder_name) / "data" / "learned_without.xyz"
            if without_path.exists():
                data_sources['without'] = {
                    'df': pd.read_csv(without_path, nrows=args.plot_steps),
                    'title': 'Learned Without'
                }
        except Exception as e:
            print(f"Warning: Could not load without learner data: {e}")
    
    methods = data_sources
    frames, titles = get_frames_and_titles()
    
    if not frames:
        print("Warning: No data loaded. Specify at least one data source (--dataset, --GT, etc.)")
    else:
        print(f"Loaded {len(frames)} data sources: {titles}")
    
    # Clean log file
    if args.export and not args.no_clean:
        log_path = Path(args.folder_name) / "log.txt"
        if log_path.exists():
            print("Cleaning log file")
            log_path.unlink()
    
    # Execute plotting commands
    if args.plot_m:
        plot_fields_from_dataframes(["mx", "my", "mz"])
    
    if args.plot_r:
        plot_fields_from_dataframes(["rx", "ry", "rz"])
    
    if args.plot_field != "None":
        plot_fields_from_dataframes([args.plot_field])
    
    if args.plot_msq:
        plot_fields_from_dataframes(["sqm"])
    
    if args.plot_rsq:
        plot_fields_from_dataframes(["sqr"])
    
    if args.plot_first != "None":
        plot_first(args.plot_first)
    
    if args.plot_first_mx:
        plot_first("mx")
    
    if args.plot_first_rx:
        plot_first("rx")
    
    if args.plot_compatibility:
        plot_compatibility_errors()
    
    if args.plot_L_errors:
        plot_L_errors()
    
    if args.plot_RB_errors:
        plot_fields_errors(["mx", "my", "mz"], field_name="m")
    
    if args.plot_HT_errors:
        plot_fields_errors(["mx", "my", "mz"], field_name="m")
        plot_fields_errors(["rx", "ry", "rz"], field_name="r")
    
    if args.plot_Es:
        plot_energy_models()
    
    if args.plot_Ls:
        plot_l_models()
    
    if args.plot_training_errors:
        comparison.plot_training_errors(args)
    
    if args.general_learner:
        print(f"GeneralSystemLearner support planned for: {args.general_learner}")


if __name__ == "__main__":
    main()
