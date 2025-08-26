import numpy as np
import matplotlib.pyplot as plt

def get_col_hist_grid(*arrays, bins=30, nrows=1, ncols=None, figsize=(5,4),
                      col_labs=None, plot_labs=None, hist_kwargs=None):
    """
    Given multiple (n, d) arrays (same d), return one matplotlib Figure per column.
    Each Figure contains one histogram per array (for that column).

    Parameters:
        *arrays: arbitrary number of numpy arrays, each shape (n, d)
        bins: number of histogram bins (default=30)
        hist_kwargs: dict of extra kwargs for plt.hist (optional)
        col_labs: list of d names associated with the columns.
        plot_labs: list of len(arrays) names associated with the arrays.

    Returns:
        figs: list of matplotlib Figure objects (len=d)
    """

    n_cols = arrays[0].shape[1]
    if not all(a.shape[1] == n_cols for a in arrays):
        raise ValueError("All arrays must have the same number of columns")

    if hist_kwargs is None:
        hist_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_cols / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_cols)]
    if plot_labs is None:
        plot_labs = [f"Array {i+1}" for i in range(len(arrays))]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_cols):
        ax = axs[col]
        for idx, arr in enumerate(arrays):
            ax.hist(arr[:, col], bins=bins, alpha=0.5, label=plot_labs[idx], **hist_kwargs)
        ax.set_title(col_labs[col])
        ax.legend()

    # Hide unused axes
    for k in range(n_cols, nrows*ncols):
        fig.delaxes(axs[k])
    return fig


def plot_trace(samp_arr, nrows=1, ncols=None, figsize=(5,4),
               col_labs=None, plot_kwargs=None):
    """
    Generate one trace plot per column of `samp_arr`.

    Parameters:
        arr: numpy array of shape (n, d)
        col_labs: list of labels associated with each column of `samp_arr`.

    Returns:
        fig: matplotlib Figure object
        ax: matplotlib Axes object
    """
    n_itr, n_cols = samp_arr.shape
    x = np.arange(n_itr)

    if plot_kwargs is None:
        plot_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_cols / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_cols)]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_cols):
        ax = axs[col]
        ax.plot(x, samp_arr[:,col], **plot_kwargs)
        ax.set_title(col_labs[col])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    # Hide unused axes
    for k in range(n_cols, nrows*ncols):
        fig.delaxes(axs[k])

    return fig
