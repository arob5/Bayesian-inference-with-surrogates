# experiments/vsem/rkpcn_analysis/plots.py
"""
Plotting utilities for RKPCN single-replicate analysis.

All plot functions return (fig, axes) and do not call plt.show(),
so the notebook controls display.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from uncprop.utils.grid import Grid, normalize_density_over_grid


# ---------------------------------------------------------------------------
# GP surrogate visualization
# ---------------------------------------------------------------------------

def plot_gp_predictive(rep, grid=None, figsize=(14, 5)):
    """Plot GP predictive mean and standard deviation heatmaps.

    Also overlays design points and (if clipped GP) the clipping bound.

    Args:
        rep: VSEMReplicate with posterior_surrogate and grid.
        grid: Grid object. Uses rep.grid if None.

    Returns:
        (fig, axes) tuple.
    """
    if grid is None:
        grid = rep.grid

    pred = rep.posterior_surrogate.surrogate(grid.flat_grid)
    pred_mean = np.array(pred.mean.ravel())
    pred_std = np.array(pred.stdev.ravel())
    n = grid.n_points_per_dim

    xx = np.array(grid.flat_grid[:, 0]).reshape(n[1], n[0])
    yy = np.array(grid.flat_grid[:, 1]).reshape(n[1], n[0])

    design_x = np.array(rep.design.X)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Predictive mean
    ax = axes[0]
    im = ax.pcolormesh(xx, yy, pred_mean.reshape(n[1], n[0]), shading='auto')
    ax.scatter(design_x[:, 0], design_x[:, 1], c='red', s=50,
               edgecolors='white', zorder=5, label='design')
    ax.set_title('GP predictive mean')
    ax.set_xlabel(grid.dim_names[0])
    ax.set_ylabel(grid.dim_names[1])
    ax.legend()
    plt.colorbar(im, ax=ax)

    # Predictive std
    ax = axes[1]
    im = ax.pcolormesh(xx, yy, pred_std.reshape(n[1], n[0]), shading='auto',
                        cmap='Reds')
    ax.scatter(design_x[:, 0], design_x[:, 1], c='blue', s=50,
               edgecolors='white', zorder=5, label='design')
    ax.set_title('GP predictive std')
    ax.set_xlabel(grid.dim_names[0])
    ax.set_ylabel(grid.dim_names[1])
    ax.legend()
    plt.colorbar(im, ax=ax)

    # Predictive coefficient of variation (std / |mean|)
    ax = axes[2]
    cv = pred_std / (np.abs(pred_mean) + 1e-10)
    im = ax.pcolormesh(xx, yy, cv.reshape(n[1], n[0]), shading='auto',
                        cmap='Oranges')
    ax.scatter(design_x[:, 0], design_x[:, 1], c='blue', s=50,
               edgecolors='white', zorder=5, label='design')
    ax.set_title('GP predictive CV (std/|mean|)')
    ax.set_xlabel(grid.dim_names[0])
    ax.set_ylabel(grid.dim_names[1])
    ax.legend()
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Density heatmaps
# ---------------------------------------------------------------------------

def plot_density_heatmaps(grid_densities, grid, names=None, figsize_per=4):
    """Plot normalized log-density heatmaps for multiple distributions.

    Args:
        grid_densities: dict mapping name -> (n_grid,) log-density array.
        grid: Grid object.
        names: list of names to plot. Defaults to all keys.
        figsize_per: width per subplot.

    Returns:
        (fig, axes).
    """
    import jax.numpy as jnp

    if names is None:
        names = list(grid_densities.keys())

    n = grid.n_points_per_dim
    xx = np.array(grid.flat_grid[:, 0]).reshape(n[1], n[0])
    yy = np.array(grid.flat_grid[:, 1]).reshape(n[1], n[0])

    fig, axes = plt.subplots(1, len(names),
                              figsize=(figsize_per * len(names), figsize_per))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        logp = grid_densities[name]
        logp_norm = normalize_density_over_grid(
            logp, cell_area=grid.cell_area)[0].squeeze()
        p = np.array(np.exp(np.array(logp_norm)))

        im = ax.pcolormesh(xx, yy, p.reshape(n[1], n[0]), shading='auto')
        ax.set_title(name)
        ax.set_xlabel(grid.dim_names[0])
        ax.set_ylabel(grid.dim_names[1])
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------------

def plot_traces(results, par_names=None, figsize=(14, 3)):
    """Plot log-density and u-parameter traces for multiple variants.

    Args:
        results: dict mapping label -> result dict.
        par_names: parameter names.
        figsize: (width, height) per row.

    Returns:
        list of (fig, axes).
    """
    figs = []

    # Log-density traces (post-burnin)
    n_variants = len(results)
    fig, axes = plt.subplots(n_variants, 1,
                              figsize=(figsize[0], figsize[1] * n_variants),
                              sharex=True, squeeze=False)
    for i, (label, res) in enumerate(results.items()):
        ax = axes[i, 0]
        ld = res['logdensities'][res['n_burnin']:]
        ax.plot(ld, linewidth=0.3, alpha=0.7)
        ax.set_ylabel(f'{label}')
        ax.set_title(f'Log-density trace — {label}', fontsize=10)
    axes[-1, 0].set_xlabel('Iteration (post-burnin)')
    fig.tight_layout()
    figs.append(fig)

    # u-parameter traces
    first = next(iter(results.values()))
    d = first['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{i+1}' for i in range(d)]

    for d_idx, pname in enumerate(par_names):
        fig, axes = plt.subplots(n_variants, 1,
                                  figsize=(figsize[0], figsize[1] * n_variants),
                                  sharex=True, squeeze=False)
        fig.suptitle(f'Trace: {pname}', fontsize=12, y=1.01)
        for i, (label, res) in enumerate(results.items()):
            ax = axes[i, 0]
            pos = res['positions'][res['n_burnin']:, d_idx]
            ax.plot(pos, linewidth=0.3, alpha=0.7)
            ax.set_ylabel(label)
        axes[-1, 0].set_xlabel('Iteration (post-burnin)')
        fig.tight_layout()
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Autocorrelation plots
# ---------------------------------------------------------------------------

def plot_acf(results, par_names=None, max_lag=2000, figsize=(14, 5)):
    """Plot autocorrelation functions for u-parameters and log-density.

    Args:
        results: dict mapping label -> result dict.
        par_names: parameter names.
        max_lag: maximum lag.

    Returns:
        (fig, axes).
    """
    from .diagnostics import autocorrelation

    first = next(iter(results.values()))
    d = first['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{i+1}' for i in range(d)]

    fig, axes = plt.subplots(1, d + 1, figsize=figsize)

    # u-parameter ACFs
    for d_idx in range(d):
        ax = axes[d_idx]
        for label, res in results.items():
            pos = res['positions'][res['n_burnin']:, d_idx]
            acf = autocorrelation(pos, max_lag=max_lag)
            ax.plot(acf, label=label, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(f'ACF: {par_names[d_idx]}')
        ax.set_xlabel('Lag')
        if d_idx == 0:
            ax.set_ylabel('Autocorrelation')
        ax.legend(fontsize=8)

    # Log-density ACF
    ax = axes[d]
    for label, res in results.items():
        ld = res['logdensities'][res['n_burnin']:]
        acf = autocorrelation(ld, max_lag=max_lag)
        ax.plot(acf, label=label, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('ACF: log-density')
    ax.set_xlabel('Lag')
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Scatter / contour comparison plots
# ---------------------------------------------------------------------------

def plot_samples_vs_ep(results, ep_density, grid, thin=5,
                       reference_samples=None, figsize_per=5):
    """Scatter plot of RKPCN samples overlaid on EP density contours.

    Each variant gets its own panel. EP density is shown as filled contours
    in the background.

    Args:
        results: dict mapping label -> result dict.
        ep_density: (n_grid,) log EP density on grid (unnormalized).
        grid: Grid object.
        thin: thinning factor for scatter points.
        reference_samples: optional dict of {name: (n, d) array} to plot
            in additional panels (e.g., exact, mean, eup samples).
        figsize_per: size per panel.

    Returns:
        (fig, axes).
    """
    import jax.numpy as jnp

    n = grid.n_points_per_dim
    xx = np.array(grid.flat_grid[:, 0]).reshape(n[1], n[0])
    yy = np.array(grid.flat_grid[:, 1]).reshape(n[1], n[0])

    logp_norm = normalize_density_over_grid(
        ep_density, cell_area=grid.cell_area)[0].squeeze()
    p_ep = np.array(np.exp(np.array(logp_norm))).reshape(n[1], n[0])

    # Determine number of panels
    ref_names = list(reference_samples.keys()) if reference_samples else []
    all_labels = ref_names + list(results.keys())
    n_panels = len(all_labels)
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per * ncols, figsize_per * nrows),
                              squeeze=False)

    for idx, label in enumerate(all_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        # EP contours
        ax.contourf(xx, yy, p_ep, levels=15, alpha=0.5, cmap='Blues')

        # Scatter
        if label in (reference_samples or {}):
            samp = np.array(reference_samples[label][::thin])
            color = 'green'
        else:
            samp = results[label]['post_burnin'][::thin]
            color = 'red'

        ax.scatter(samp[:, 0], samp[:, 1], alpha=0.3, s=5, c=color)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(grid.dim_names[0])
        if c == 0:
            ax.set_ylabel(grid.dim_names[1])

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig, axes
