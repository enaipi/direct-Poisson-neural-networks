"""Data loading and processing utilities for plot_compare."""

import pandas as pd
import numpy as np
import torch
from pathlib import Path


def load_dataframes(folder_name, plot_steps=None, methods_to_load=None):
    """
    Load all available dataframes from folder.
    
    Args:
        folder_name: Path to folder with data files
        plot_steps: Maximum rows to read (None = all)
        methods_to_load: Dict with 'without', 'soft', 'implicit', 'GT', 'dataset' booleans
    
    Returns:
        Dict of {method_name: (dataframe, title_name)}
    """
    if methods_to_load is None:
        methods_to_load = {'without': True, 'soft': True, 'implicit': True, 'GT': False, 'dataset': True}
    
    dataframes = {}
    file_mapping = {
        'dataset': ('data/dataset.xyz', 'Dataset'),
        'GT': ('data/generalization.xyz', 'Ground Truth'),
        'implicit': ('data/learned_implicit.xyz', 'Learned Implicit'),
        'soft': ('data/learned_soft.xyz', 'Learned Soft'),
        'without': ('data/learned_without.xyz', 'Learned Without'),
    }
    
    for method, (file_suffix, title) in file_mapping.items():
        if not methods_to_load.get(method, False):
            continue
        
        file_path = Path(folder_name) / file_suffix
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, nrows=plot_steps)
                dataframes[method] = {'df': df, 'title': title}
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    return dataframes


def remove_returns(xs, ys):
    """Remove points that have earlier timestep than their predecessor."""
    print(len(xs))
    newxs = [xs[0]]
    newys = [ys[0]]
    for i in range(1, len(xs)):
        if xs[i] > xs[i-1]:
            newxs.append(xs[i])
            newys.append(ys[i])
    return newxs, newys


def filter_data(xs, ys, plot_every=1):
    """Filter x and y arrays by keeping every nth element."""
    newxs = [xs[0]]
    newys = [ys[0]]
    for i in range(1, len(xs)):
        if i % plot_every == 0:
            newxs.append(xs[i])
            newys.append(ys[i])
    return np.array(newxs), np.array(newys)


def sort_data(xs, ys):
    """Sort two arrays based on values in xs."""
    p = np.argsort(xs)
    newxs = xs[p]
    newys = ys[p]
    return newxs, newys


def split_data_to_forward_paths(xs, ys):
    """
    Split data into forward paths based on time (x) values.
    When xs[i] < xs[i-1], a new path starts.
    """
    if len(xs) != len(ys):
        raise Exception("Length of xs and ys not equal.")
    paths = [[ys[0]]]
    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:  # return to time zero, new path
            paths.append([ys[i]])
        else:
            paths[len(paths)-1].append(ys[i])  # add value to current path
    return paths


def split_to_forward_paths(df, field):
    """Return array of forward paths for dataframe df and a given field."""
    ts = df["time"]
    values = df[field]
    return split_data_to_forward_paths(ts, values)


def reject_outliers(data, m=3):
    """Remove outliers from data using mean ± m*std threshold."""
    result = data[abs(data - np.mean(data)) < m * np.std(data)]
    rejected = len(data) - len(result)
    if rejected > 0:
        print(f"Warning: Rejecting {rejected} out of {len(data)}")
    return result


def normalize(Ls):
    """Normalize L matrices to canonical form based on dimension."""
    dim = len(Ls[0])
    print(f"Normalization to canonical L for dimension {dim}")
    normalization = np.linalg.norm(Ls) / np.sqrt(len(Ls) / dim)
    return Ls * dim / normalization


def load_normalized_Ls(data_frame, dim):
    """
    Load and construct L matrices from dataframe columns.
    
    For dim=3: L_01, L_02, L_12
    For dim=4: L_01, L_02, L_03, L_12, L_13, L_23
    For dim=6: All 15 independent components
    """
    if dim == 3:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_12s = data_frame["L_12"]
        
        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i]],
             [-L_01s[i], 0.0, L_12s[i]],
             [-L_02s[i], -L_12s[i], 0.0]]
            for i in range(len(L_01s))
        ])
    
    elif dim == 4:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_03s = data_frame["L_03"]
        L_12s = data_frame["L_12"]
        L_13s = data_frame["L_13"]
        L_23s = data_frame["L_23"]
        
        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i], L_03s[i]],
             [-L_01s[i], 0.0, L_12s[i], L_13s[i]],
             [-L_02s[i], -L_12s[i], 0.0, L_23s[i]],
             [-L_03s[i], -L_13s[i], -L_23s[i], 0.0]]
            for i in range(len(L_01s))
        ])
    
    elif dim == 6:
        L_01s = data_frame["L_01"]
        L_02s = data_frame["L_02"]
        L_03s = data_frame["L_03"]
        L_04s = data_frame["L_04"]
        L_05s = data_frame["L_05"]
        L_12s = data_frame["L_12"]
        L_13s = data_frame["L_13"]
        L_14s = data_frame["L_14"]
        L_15s = data_frame["L_15"]
        L_23s = data_frame["L_23"]
        L_24s = data_frame["L_24"]
        L_25s = data_frame["L_25"]
        L_34s = data_frame["L_34"]
        L_35s = data_frame["L_35"]
        L_45s = data_frame["L_45"]
        
        Ls = np.array([
            [[0.0, L_01s[i], L_02s[i], L_03s[i], L_04s[i], L_05s[i]],
             [-L_01s[i], 0.0, L_12s[i], L_13s[i], L_14s[i], L_15s[i]],
             [-L_02s[i], -L_12s[i], 0.0, L_23s[i], L_24s[i], L_25s[i]],
             [-L_03s[i], -L_13s[i], -L_23s[i], 0.0, L_34s[i], L_35s[i]],
             [-L_04s[i], -L_14s[i], -L_24s[i], -L_34s[i], 0.0, L_45s[i]],
             [-L_05s[i], -L_15s[i], -L_25s[i], -L_35s[i], -L_45s[i], 0.0]]
            for i in range(len(L_01s))
        ])
    
    else:
        raise Exception(f"Dimension {dim} not implemented")
    
    # Ensure antisymmetry
    Ls -= np.transpose(Ls, (0, 2, 1))
    Ls = normalize(Ls)
    return Ls
