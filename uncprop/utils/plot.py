import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

def set_plot_theme():
    sns.set_theme(style='white', palette='colorblind')
    sns.set_context("paper", font_scale=1.5)

    # Specific Paul Tol color scheme when comparing different posteriors
    colors = {
        'exact': "#4477AA",
        'mean': "#EE6677",
        'eup': "#228833",
        'ep': "#CCBB44",
        'aux': "#888888"
    }

    return colors


def smart_subplots(
    nrows=None, ncols=None, figsize=None,
    tight=True, nplots=None, max_cols=None,
    flatten=True, squeeze=True, sharex=False, sharey=False,
    **kwargs
):
    """
    Wrapper for plt.subplots with enhanced features.
    - tight: Use tight_layout if True
    - nrows, ncols, figsize: as in plt.subplots
    - nplots, max_cols: alternative way to choose grid shape (row-major order)
    - flatten: If True (default), always return flat list of axes
    """
 
    # grid specification logic
    if nplots is not None:
        if (nrows is not None) or (ncols is not None):
            raise ValueError("Specify either nplots (with optional max_cols), OR nrows/ncols, not both.")
        if nplots <= 0:
            raise ValueError("nplots must be positive")
        if max_cols is not None and max_cols <= 0:
            raise ValueError("max_cols must be positive if specified")
        
        # row major logic: Fill rows first
        _max_cols = max_cols if max_cols is not None else 1
        ncols = min(nplots, _max_cols)
        nrows = math.ceil(nplots / ncols)
    elif nrows is not None or ncols is not None:
        if nrows is None or ncols is None:
            raise ValueError("If specifying nrows or ncols, both must be provided.")
        if nrows <= 0 or ncols <= 0:
            raise ValueError("nrows and ncols must both be positive")
        nplots = nrows * ncols
    else:
        # Defaults to one plot
        nrows, ncols, nplots = 1, 1, 1

    # Reasonable auto-sizing if not specified
    if figsize is None:
        base_width = 4.0  # inches per subplot
        base_height = 3.0
        width = ncols * base_width
        height = nrows * base_height
        figsize = (width, height)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=squeeze,
        sharex=sharex,
        sharey=sharey,
        **kwargs
    )

    if tight:
        fig.tight_layout()

    if flatten:
        # Always return as flat list of axes (even if originally an array or single Axes)
        if isinstance(axs, np.ndarray):
            out_axs = axs.flatten().tolist()
        else:
            out_axs = [axs]
        return fig, out_axs
    else:
        return fig, axs