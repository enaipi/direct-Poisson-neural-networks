"""Plotting and visualization functions."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def add_plot(axes, xs, ys, name="", split=True):
    """
    Add a line plot to axes.
    
    Args:
        axes: matplotlib axes object
        xs: x values (time or iterations)
        ys: y values (errors or other quantities)
        name: Legend label
        split: Whether to split data into forward paths
    """
    if split:
        paths = split_to_paths(xs, ys)
        for path in paths:
            axes.plot(path, label=name, alpha=0.5)
    else:
        axes.plot(ys, label=name)


def plot_field(dataframes_dict, field, plot_every=1):
    """
    Plot a field across multiple dataframes.
    
    Args:
        dataframes_dict: Dict of {method: {'df': dataframe, 'title': title}}
        field: Field name to plot (column in dataframe)
        plot_every: Plot every nth point
    """
    plt.figure()
    for method, info in dataframes_dict.items():
        df = info['df']
        title = info['title']
        
        if field not in df.columns:
            print(f"Warning: Field {field} not in {method}")
            continue
        
        ts = df["time"].values
        values = df[field].values
        
        if plot_every > 1:
            ts = ts[::plot_every]
            values = values[::plot_every]
        
        plt.plot(ts, values, label=title, alpha=0.7)
    
    plt.xlabel("Time")
    plt.ylabel(field)
    plt.legend()
    plt.title(f"Field: {field}")
    plt.show()


def plot_first(field):
    """Plot only first time point for a field."""
    # This function requires loading data separately
    # Placeholder for now
    print(f"Plotting first point of {field}")


def split_to_paths(xs, ys):
    """
    Split data into forward paths based on time values.
    When xs[i] < xs[i-1], a new path starts.
    """
    if len(xs) != len(ys):
        raise Exception("Length of xs and ys not equal.")
    
    paths = [[ys[0]]]
    for i in range(1, len(xs)):
        if xs[i] < xs[i-1]:  # return to time zero
            paths.append([ys[i]])
        else:
            paths[-1].append(ys[i])
    
    return paths


def plot_errors_line(axes, iterations, errors, label="", color=None):
    """
    Plot errors as a line.
    
    Args:
        axes: matplotlib axes
        iterations: x values
        errors: y values (errors)
        label: Legend label
        color: Line color
    """
    if color:
        axes.plot(iterations, errors, label=label, color=color)
    else:
        axes.plot(iterations, errors, label=label)


def plot_errors_scatter(axes, xs, errors, label="", marker="o"):
    """
    Plot errors as scatter points.
    
    Args:
        axes: matplotlib axes
        xs: x values (e.g., state variables)
        errors: y values (errors)
        label: Legend label
        marker: Marker style
    """
    axes.scatter(xs, errors, label=label, marker=marker, alpha=0.6)


def add_log(filename, value):
    """
    Append value to log file.
    
    Args:
        filename: Log file path
        value: Value to log
    """
    try:
        with open(filename + ".log", "a") as f:
            f.write(f"{value}\n")
    except Exception as e:
        print(f"Warning: Could not write to log {filename}: {e}")


def format_title(name, model_type):
    """Format a title string with model information."""
    return f"{name} ({model_type})"


def plot_3d_surface(fig, ax, X, Y, Z, xlabel="", ylabel="", zlabel="", title=""):
    """
    Plot a 3D surface.
    
    Args:
        fig: matplotlib figure
        ax: 3D axes
        X, Y, Z: Grid arrays
        xlabel, ylabel, zlabel: Axis labels
        title: Plot title
    """
    from matplotlib import cm
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


def plot_3d_scatter(ax, xs, ys, zs, label="", marker="o"):
    """
    Plot 3D scatter points.
    
    Args:
        ax: 3D axes
        xs, ys, zs: Coordinate arrays
        label: Legend label
        marker: Marker style
    """
    ax.scatter(xs, ys, zs, label=label, marker=marker, alpha=0.6)


def compare_two_methods(df1, df2, field, title1="Method 1", title2="Method 2"):
    """
    Compare a field between two methods side-by-side.
    
    Args:
        df1: First dataframe
        df2: Second dataframe
        field: Field to compare
        title1: Title for first plot
        title2: Title for second plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if field in df1.columns:
        ax1.plot(df1["time"], df1[field])
        ax1.set_title(title1)
        ax1.set_xlabel("Time")
        ax1.set_ylabel(field)
    
    if field in df2.columns:
        ax2.plot(df2["time"], df2[field])
        ax2.set_title(title2)
        ax2.set_xlabel("Time")
        ax2.set_ylabel(field)
    
    plt.tight_layout()
    plt.show()
