# experiments/vsem/plots.py
"""
VSEM-specific benchmark plots.

These plots assume a 2D parameter space with a grid-based EP reference
density available. They are loaded by
``experiments/benchmarks/benchmark.py`` only when analyzing a VSEM
replicate (detected via :func:`experiments.benchmarks.replicate_loader.has_grid`).

Plots provided:

- ``plot_samples_vs_ep``          : pooled samples overlaid on EP contours
- ``plot_samples_vs_ep_annotated``: per-chain coloring + init-position stars
- ``plot_density_heatmaps``       : heatmaps of grid densities
                                    (exact/mean/eup/ep)
- ``plot_gp_predictive``          : GP predictive mean / std / CV
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from uncprop.utils.grid import normalize_density_over_grid


# =============================================================================
# GP predictive
# =============================================================================

def plot_gp_predictive(rep, grid=None, figsize=(14, 5)):
    """GP predictive mean / std / CV with design overlays."""
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

    for ax, (data, title, cmap, design_color) in zip(
            axes,
            [(pred_mean, 'GP predictive mean', None, 'red'),
             (pred_std, 'GP predictive std', 'Reds', 'blue'),
             (pred_std / (np.abs(pred_mean) + 1e-10),
              'GP predictive CV (std/|mean|)', 'Oranges', 'blue')]):
        im = ax.pcolormesh(xx, yy, data.reshape(n[1], n[0]),
                           shading='auto', cmap=cmap)
        ax.scatter(design_x[:, 0], design_x[:, 1], c=design_color, s=50,
                   edgecolors='white', zorder=5, label='design')
        ax.set_title(title)
        ax.set_xlabel(grid.dim_names[0])
        ax.set_ylabel(grid.dim_names[1])
        ax.legend()
        plt.colorbar(im, ax=ax)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Density heatmaps
# =============================================================================

def plot_density_heatmaps(grid_densities: dict, grid, names=None,
                          figsize_per: float = 4.5):
    """Heatmaps of the named log-densities on the grid.

    Parameters
    ----------
    grid_densities : dict
        Name -> (n_grid,) log-density array (unnormalized).
    grid : Grid
    names : list[str] or None
        Which densities to show. Defaults to all keys.
    figsize_per : float
        Figure size per panel.
    """
    if names is None:
        names = list(grid_densities.keys())

    n = grid.n_points_per_dim
    xx = np.array(grid.flat_grid[:, 0]).reshape(n[1], n[0])
    yy = np.array(grid.flat_grid[:, 1]).reshape(n[1], n[0])

    fig, axes = plt.subplots(1, len(names),
                             figsize=(figsize_per * len(names), figsize_per),
                             squeeze=False)
    axes = axes[0]

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


# =============================================================================
# Scatter vs. EP contours
# =============================================================================

def _setup_ep_contours(ep_density, grid):
    n = grid.n_points_per_dim
    xx = np.array(grid.flat_grid[:, 0]).reshape(n[1], n[0])
    yy = np.array(grid.flat_grid[:, 1]).reshape(n[1], n[0])
    logp_norm = normalize_density_over_grid(
        ep_density, cell_area=grid.cell_area)[0].squeeze()
    p_ep = np.array(np.exp(np.array(logp_norm))).reshape(n[1], n[0])
    return xx, yy, p_ep


def plot_samples_vs_ep(results, ep_density, grid, thin: int = 5,
                       reference_samples=None, figsize_per: float = 5.0):
    """Pooled samples overlaid on EP contours (one panel per variant)."""
    xx, yy, p_ep = _setup_ep_contours(ep_density, grid)

    ref_names = list(reference_samples.keys()) if reference_samples else []
    all_labels = ref_names + list(results.keys())
    n = len(all_labels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols,
                                      figsize_per * nrows),
                             squeeze=False)

    for idx, label in enumerate(all_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.contourf(xx, yy, p_ep, levels=15, alpha=0.5, cmap='Blues')

        if label in (reference_samples or {}):
            samp = np.asarray(reference_samples[label])[::thin]
            ax.scatter(samp[:, 0], samp[:, 1], alpha=0.3, s=5, c='green')
        else:
            samp = results[label]['post_burnin'][::thin]
            ax.scatter(samp[:, 0], samp[:, 1], alpha=0.3, s=5, c='red')

        ax.set_title(label, fontsize=11)
        ax.set_xlabel(grid.dim_names[0])
        if c == 0:
            ax.set_ylabel(grid.dim_names[1])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_samples_vs_ep_annotated(results, ep_density, grid, thin: int = 5,
                                 reference_samples=None,
                                 figsize_per: float = 5.0):
    """Per-chain colored scatter vs EP with init-position stars.

    For multi-chain variants, colors by chain and uses transparency
    proportional to per-chain effective weight
    (``mode_weight / n_chains_in_mode``). Failed chains are gray.
    """
    xx, yy, p_ep = _setup_ep_contours(ep_density, grid)

    ref_names = list(reference_samples.keys()) if reference_samples else []
    all_labels = ref_names + list(results.keys())
    n = len(all_labels)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols,
                                      figsize_per * nrows),
                             squeeze=False)

    for idx, label in enumerate(all_labels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.contourf(xx, yy, p_ep, levels=15, alpha=0.5, cmap='Blues')

        if label in (reference_samples or {}):
            samp = np.asarray(reference_samples[label])[::thin]
            ax.scatter(samp[:, 0], samp[:, 1], alpha=0.3, s=5, c='green')
        else:
            res = results[label]
            per_chain = res.get('per_chain_results')
            mode_weights = res.get('mode_weights')
            mode_labels = res.get('mode_labels')

            if per_chain is not None and len(per_chain) > 1:
                n_ch = len(per_chain)
                chain_colors = plt.cm.Set1(np.linspace(0, 1, min(n_ch, 9)))

                # Per-chain effective weight (mode_weight / n_in_mode)
                chain_weights = np.zeros(n_ch)
                if mode_weights is not None and mode_labels is not None:
                    mla = np.asarray(mode_labels)
                    for k, w_k in enumerate(np.asarray(mode_weights)):
                        members = np.where(mla == k)[0]
                        if len(members) > 0:
                            for m in members:
                                chain_weights[m] = float(w_k) / len(members)
                else:
                    chain_weights[:] = 1.0 / n_ch

                for m, cr in enumerate(per_chain):
                    cs = cr['post_burnin'][::thin]
                    w = chain_weights[m]
                    mode_id = (int(mode_labels[m])
                               if mode_labels is not None else -1)
                    mode_str = f'm{mode_id}' if mode_id >= 0 else 'FAIL'
                    alpha = max(0.1, min(0.6, w * n_ch * 0.3))
                    ax.scatter(cs[:, 0], cs[:, 1], alpha=alpha, s=5,
                               c=[chain_colors[m % len(chain_colors)]],
                               label=f'ch{m} [{mode_str}, w={w:.2f}]')

                init_pos = res.get('init_positions')
                if init_pos is not None:
                    ip = np.array(init_pos)
                    for m in range(min(len(ip), n_ch)):
                        ax.scatter(ip[m, 0], ip[m, 1], marker='*', s=250,
                                   c=[chain_colors[m % len(chain_colors)]],
                                   edgecolors='black', zorder=10,
                                   linewidths=1.0)

                ax.legend(fontsize=6, loc='lower right', markerscale=2)
            else:
                samp = res['post_burnin'][::thin]
                ax.scatter(samp[:, 0], samp[:, 1], alpha=0.3, s=5, c='red')

        ax.set_title(label, fontsize=11)
        ax.set_xlabel(grid.dim_names[0])
        if c == 0:
            ax.set_ylabel(grid.dim_names[1])

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.tight_layout()
    return fig, axes
