# experiments/benchmarks/plots.py
"""
Generic plotting for benchmark analysis.

All plot functions here are experiment-agnostic — they only need the
post-burnin samples and (optionally) multi-chain metadata (per-chain
samples, mode labels, init positions, sample weights). Experiment-
specific plots (e.g. contour overlays on a 2D EP grid) live in the
respective experiment's ``plots.py`` and are loaded conditionally by
:mod:`experiments.benchmarks.benchmark`.

Core plots produced per variant:
    - corner : univariate marginals on the diagonal, pairwise 2D
               marginals on the off-diagonals
    - traces : per-chain u-position traces (multi-chain runs only)
    - acf    : autocorrelation of log-density post-burnin

Plus one cross-variant overview:
    - marginals_overview : all variants' univariate marginals overlaid
"""
from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from uncprop.utils.diagnostics import compute_ess


# Color palette for chain / mode assignment. Extends naturally beyond len.
CHAIN_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]


# =============================================================================
# Corner plot (univariate diagonals + pairwise off-diagonals)
# =============================================================================

def plot_corner(
    post_burnin: np.ndarray,
    sample_weights: np.ndarray | None = None,
    per_chain_post_burnin: list[np.ndarray] | None = None,
    mode_labels: np.ndarray | None = None,
    init_positions: np.ndarray | None = None,
    par_names: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    thin: int = 1,
    scatter_alpha: float = 0.3,
    scatter_size: float = 4.0,
):
    """Corner plot of samples (univariate on diagonal, pairwise off-diagonal).

    When ``per_chain_post_burnin`` and ``mode_labels`` are provided,
    off-diagonals color scatters by mode and diagonals draw one KDE
    curve per mode (plus the pooled histogram in gray). Without
    multi-chain metadata, diagonals draw a single histogram and
    off-diagonals a single scatter.

    Parameters
    ----------
    post_burnin : (N, d) array
        Pooled samples from the variant.
    sample_weights : (N,) array or None
        Per-sample weights summing to 1; if provided, used in the
        pooled diagonal histograms.
    per_chain_post_burnin : list of arrays or None
        Per-chain post-burnin samples (for mode coloring and per-mode
        diagonal KDEs). Combined with ``mode_labels``.
    mode_labels : (n_chains,) int array or None
        Mode assignment per chain (``-1`` = failed).
    init_positions : (n_chains, d) array or None
        Starting position per chain, rendered as star markers.
    par_names : list[str] or None
    title : str or None
    figsize : tuple or None
        Defaults to ``(2.5 * d, 2.5 * d)``.
    thin : int
        Thin applied to off-diagonal scatters.
    scatter_alpha, scatter_size : styling.

    Returns
    -------
    (fig, axes) with ``axes`` shape ``(d, d)``.
    """
    d = post_burnin.shape[1]
    if par_names is None:
        par_names = [f'u{i}' for i in range(d)]
    if figsize is None:
        figsize = (2.5 * d, 2.5 * d)

    fig, axes = plt.subplots(d, d, figsize=figsize, squeeze=False)

    # Build per-mode groupings if multi-chain data is available
    have_modes = (per_chain_post_burnin is not None
                  and mode_labels is not None
                  and len(per_chain_post_burnin) == len(mode_labels))

    mode_chains: dict[int, list[np.ndarray]] = {}
    chain_colors: dict[int, str] = {}
    if have_modes:
        for m, (chain_samp, lbl) in enumerate(
                zip(per_chain_post_burnin, mode_labels)):
            lbl = int(lbl)
            mode_chains.setdefault(lbl, []).append(np.asarray(chain_samp))
            # Color each chain by its mode (failed chains = gray)
            if lbl < 0:
                chain_colors[m] = '#cccccc'
            else:
                chain_colors[m] = CHAIN_COLORS[lbl % len(CHAIN_COLORS)]

    # ---- Diagonals ----
    for i in range(d):
        ax = axes[i, i]
        # Pooled histogram (gray background)
        ax.hist(post_burnin[:, i], bins=40,
                weights=sample_weights, color='#cccccc',
                edgecolor='none', density=True, label='pooled')

        # Per-mode KDE curves
        if have_modes:
            xmin = post_burnin[:, i].min()
            xmax = post_burnin[:, i].max()
            xs = np.linspace(xmin, xmax, 200)
            for lbl, chains in sorted(mode_chains.items()):
                if lbl < 0:
                    continue
                pooled_mode = np.concatenate(chains, axis=0)[:, i]
                if pooled_mode.size < 5:
                    continue
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(pooled_mode)
                    color = CHAIN_COLORS[lbl % len(CHAIN_COLORS)]
                    ax.plot(xs, kde(xs), color=color, lw=1.8,
                            label=f'mode {lbl}')
                except Exception:
                    pass

        # Init positions: vertical ticks at bottom
        if init_positions is not None:
            for m, ip in enumerate(np.asarray(init_positions)):
                color = chain_colors.get(m, 'k') if have_modes else 'k'
                ax.axvline(ip[i], color=color, lw=1.0, alpha=0.6,
                           linestyle='--')

        ax.set_ylabel('density' if i == 0 else '')
        if i < d - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(par_names[i])

    # ---- Off-diagonals ----
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            ax = axes[i, j]
            if j > i:
                # Upper triangle unused
                ax.set_visible(False)
                continue

            # i > j: row i vs col j (y = par_names[i], x = par_names[j])
            if have_modes:
                for m, chain_samp in enumerate(per_chain_post_burnin):
                    cs = np.asarray(chain_samp)
                    if cs.shape[0] == 0:
                        continue
                    cs_thin = cs[::thin]
                    ax.scatter(cs_thin[:, j], cs_thin[:, i],
                               s=scatter_size, alpha=scatter_alpha,
                               c=chain_colors.get(m, 'k'),
                               edgecolors='none')
            else:
                p = post_burnin[::thin]
                ax.scatter(p[:, j], p[:, i],
                           s=scatter_size, alpha=scatter_alpha,
                           c='#1f77b4', edgecolors='none')

            if init_positions is not None:
                ip = np.asarray(init_positions)
                for m in range(ip.shape[0]):
                    color = chain_colors.get(m, 'k') if have_modes else 'k'
                    ax.scatter(ip[m, j], ip[m, i], marker='*', s=220,
                               c=[color], edgecolors='black', zorder=10,
                               linewidth=0.7)

            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(par_names[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(par_names[i])

    if title is not None:
        fig.suptitle(title)

    # Mode legend (if applicable) — only once, at top of figure
    if have_modes:
        legend_handles = []
        for lbl in sorted(mode_chains.keys()):
            if lbl < 0:
                legend_handles.append(Line2D(
                    [0], [0], marker='o', color='w', label='failed chain',
                    markerfacecolor='#cccccc', markersize=8))
            else:
                legend_handles.append(Line2D(
                    [0], [0], marker='o', color='w', label=f'mode {lbl}',
                    markerfacecolor=CHAIN_COLORS[lbl % len(CHAIN_COLORS)],
                    markersize=8))
        if init_positions is not None:
            legend_handles.append(Line2D(
                [0], [0], marker='*', color='w', label='init',
                markerfacecolor='white',
                markeredgecolor='black', markersize=14))
        fig.legend(handles=legend_handles, loc='upper right',
                   bbox_to_anchor=(0.98, 0.98))

    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    return fig, axes


# =============================================================================
# Trace plots
# =============================================================================

def plot_traces(
    result: dict,
    par_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    thin: int = 1,
):
    """Per-dimension trace plot. Colors per chain when multi-chain data available.

    Parameters
    ----------
    result : dict
        Variant result dict. Can be single-chain (``positions`` present)
        or multi-chain (``per_chain_results`` present).
    par_names : list[str] or None
    figsize : tuple or None
    thin : int

    Returns
    -------
    (fig, axes)
    """
    # Extract per-chain traces
    per_chain = result.get('per_chain_results')
    if per_chain is not None and len(per_chain) > 0:
        chains = [np.asarray(c.get('positions', c['post_burnin']))
                  for c in per_chain]
        mode_labels = result.get('mode_labels')
    else:
        chains = [np.asarray(result['positions'])]
        mode_labels = None

    d = chains[0].shape[1]
    if par_names is None:
        par_names = [f'u{i}' for i in range(d)]
    if figsize is None:
        figsize = (10, 2.0 * d)

    fig, axes = plt.subplots(d, 1, figsize=figsize, squeeze=False)

    for i in range(d):
        ax = axes[i, 0]
        for m, chain in enumerate(chains):
            if mode_labels is not None:
                lbl = int(mode_labels[m])
                color = ('#cccccc' if lbl < 0
                         else CHAIN_COLORS[lbl % len(CHAIN_COLORS)])
            else:
                color = CHAIN_COLORS[m % len(CHAIN_COLORS)]
            ax.plot(np.arange(chain.shape[0])[::thin],
                    chain[::thin, i], color=color, lw=0.6, alpha=0.8)
        ax.set_ylabel(par_names[i])
        if i < d - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('iteration')

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Autocorrelation plots
# =============================================================================

def plot_acf(
    result: dict,
    max_lag: int = 500,
    figsize: tuple[float, float] | None = None,
):
    """Plot autocorrelation of the post-burnin log-density series.

    Uses :func:`experiments.benchmarks.diagnostics.autocorrelation`.
    """
    from experiments.benchmarks.diagnostics import autocorrelation

    ld = np.asarray(result.get('logdensities', []))
    n_burnin = int(result.get('n_burnin', 0))
    ld_post = ld[n_burnin:] if ld.size > 0 else np.empty(0)
    if ld_post.size < 2:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, 'insufficient log-density trace',
                ha='center', va='center', transform=ax.transAxes)
        return fig, [ax]

    acf = autocorrelation(ld_post, max_lag=max_lag)
    if figsize is None:
        figsize = (8, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(acf)
    ax.axhline(0, color='k', lw=0.5, linestyle='--')
    ax.set_xlabel('lag')
    ax.set_ylabel('ACF(log-density)')
    ax.set_title(result.get('label', ''))
    fig.tight_layout()
    return fig, [ax]


# =============================================================================
# Cross-variant univariate overview
# =============================================================================

def plot_marginals_overview(
    results: dict,
    par_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    max_cols: int = 3,
):
    """Overlay univariate marginals of all variants on one figure.

    Each subplot shows one dimension; each variant contributes one
    line (weighted KDE or histogram).

    Parameters
    ----------
    results : dict
        Label -> result dict with ``post_burnin`` and optional
        ``sample_weights``.
    par_names : list[str] or None
    figsize : tuple or None
    max_cols : int
    """
    if not results:
        fig, ax = plt.subplots()
        return fig, [ax]

    first = next(iter(results.values()))
    d = first['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{i}' for i in range(d)]

    ncols = min(d, max_cols)
    nrows = int(np.ceil(d / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    from scipy.stats import gaussian_kde

    for i in range(d):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        # Common x range across all variants
        xmin = min(res['post_burnin'][:, i].min() for res in results.values())
        xmax = max(res['post_burnin'][:, i].max() for res in results.values())
        xs = np.linspace(xmin, xmax, 200)

        for k, (label, res) in enumerate(results.items()):
            samp = res['post_burnin'][:, i]
            sw = res.get('sample_weights')
            if samp.size < 5:
                continue
            try:
                kde = gaussian_kde(samp, weights=sw)
                ax.plot(xs, kde(xs), label=label, lw=1.3)
            except Exception:
                pass

        ax.set_xlabel(par_names[i])
        ax.set_ylabel('density')
    # Remove unused axes
    for i in range(d, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].set_visible(False)

    # Legend in figure margin
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, axes
