"""
Per-replicate diagnostic tool for PDE experiment.

Generates trace plots, 2D marginal scatter/contour plots, and prints
acceptance rates for a specific replicate. Useful for investigating
the behavior of individual replicates (e.g., poor RKPCN mixing).

Usage:
    python diagnose_rep.py --experiment-name pde_experiment --n-design 10 --rep 0
    python diagnose_rep.py --experiment-name pde_experiment --n-design 10 --rep 0 --output-dir ./diag_plots
"""
from jax import config
config.update('jax_enable_x64', True)

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from analyze import read_samp, read_diagnostics, subdir_name


def trace_plots(samples, par_names=None, output_dir=None):
    """Trace plots for all samplers, one figure per sampler."""
    figs = []
    for name, samp in samples.items():
        samp = np.array(samp)
        n_samp, d = samp.shape
        if par_names is None:
            par_names_local = [f'u{i+1}' for i in range(d)]
        else:
            par_names_local = par_names

        fig, axs = plt.subplots(d, 1, figsize=(10, 2 * d), sharex=True)
        if d == 1:
            axs = [axs]

        for j in range(d):
            axs[j].plot(samp[:, j], linewidth=0.3)
            axs[j].set_ylabel(par_names_local[j])

        axs[0].set_title(name)
        axs[-1].set_xlabel('iteration')
        fig.tight_layout()

        if output_dir is not None:
            fig.savefig(Path(output_dir) / f'trace_{name}.pdf',
                        bbox_inches='tight')

        figs.append(fig)

    return figs


def marginal_scatter_plots(samples, dim_pairs=None, output_dir=None,
                           par_names=None):
    """
    2D marginal scatter plots comparing distributions.

    Generates one figure per pair of dimensions. Each figure overlays
    scatter plots from multiple samplers.
    """
    # Determine pairs
    first_key = next(iter(samples))
    d = samples[first_key].shape[1]
    if par_names is None:
        par_names = [f'u{i+1}' for i in range(d)]

    if dim_pairs is None:
        dim_pairs = [(i, j) for i in range(min(d, 4))
                     for j in range(i + 1, min(d, 4))]

    # Which samplers to show on each figure
    groups = [
        ('baseline', ['exact', 'mean', 'eup', 'ep_mcwmh']),
        ('rkpcn', [k for k in samples if k.startswith('rkpcn') or k == 'ep_mcwmh']),
    ]

    figs = []
    for group_name, dist_names in groups:
        available = [n for n in dist_names if n in samples]
        if len(available) == 0:
            continue

        for (di, dj) in dim_pairs:
            fig, ax = plt.subplots(figsize=(6, 5))

            for name in available:
                samp = np.array(samples[name])
                # Subsample for readability
                n_plot = min(500, samp.shape[0])
                ax.scatter(samp[:n_plot, di], samp[:n_plot, dj],
                           alpha=0.3, s=5, label=name)

            ax.set_xlabel(par_names[di])
            ax.set_ylabel(par_names[dj])
            ax.legend(markerscale=3, fontsize=8)
            ax.set_title(f'{group_name}: {par_names[di]} vs {par_names[dj]}')
            fig.tight_layout()

            if output_dir is not None:
                fig.savefig(
                    Path(output_dir) / f'scatter_{group_name}_{par_names[di]}_{par_names[dj]}.pdf',
                    bbox_inches='tight')

            figs.append(fig)

    return figs


def print_diagnostics(diagnostics):
    """Pretty-print diagnostics for a single replicate."""
    print('Diagnostics:')
    for k in sorted(diagnostics.keys()):
        print(f'  {k}: {float(diagnostics[k]):.4f}')


def main():
    parser = argparse.ArgumentParser(description='PDE per-rep diagnostics')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--n-design', type=int, required=True)
    parser.add_argument('--rep', type=int, required=True)
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save diagnostic plots')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.base_dir is None:
        base_dir = repo_root / 'out' / args.experiment_name
    else:
        base_dir = Path(args.base_dir)

    print(f'Experiment: {args.experiment_name}')
    print(f'n_design: {args.n_design}, rep: {args.rep}')
    print(f'base_dir: {base_dir}')

    # Output directory for plots
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    # Load data
    samples = read_samp(base_dir, args.n_design, args.rep)
    print(f'\nSample keys: {list(samples.keys())}')
    for k, v in samples.items():
        print(f'  {k}: shape={v.shape}')

    # Diagnostics
    try:
        diag = read_diagnostics(base_dir, args.n_design, args.rep)
        print()
        print_diagnostics(diag)
    except FileNotFoundError:
        print('\nNo diagnostics.npz found for this rep.')

    # Plots
    print('\nGenerating trace plots...')
    trace_plots(samples, output_dir=output_dir)

    print('Generating scatter plots...')
    marginal_scatter_plots(samples, output_dir=output_dir)

    if output_dir is not None:
        print(f'\nPlots saved to: {output_dir}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
