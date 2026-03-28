"""
Post-hoc analysis for VSEM experiment results.

Computes W2 distances, summarizes diagnostics, and generates plots.
Run after the experiment has completed (all reps saved to disk).

Usage:
    # From experiments/vsem/ directory:
    python analyze.py --experiment-name vsem --base-dir ../../out

    # Or import functions in a notebook:
    from analyze import compute_w2_all_setups, plot_w2_boxplots
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1'

from jax import config
config.update('jax_enable_x64', True)

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "vsem_experiment", Path(__file__).parent / "experiment.py"
)
_vsem_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vsem_exp)

read_rep_samples = _vsem_exp.read_rep_samples
read_rep_diagnostics = _vsem_exp.read_rep_diagnostics
summarize_wasserstein_reps = _vsem_exp.summarize_wasserstein_reps
summarize_diagnostics = _vsem_exp.summarize_diagnostics
check_completion_status = _vsem_exp.check_completion_status


# ---- Setup names (must match runner.py) ----
gp_tags = ['gp', 'clip_gp']
design_settings = [(4, 1e-4), (8, 1e-3), (16, 1e-2)]
setups = [(tag, n, jitter) for tag in gp_tags for n, jitter in design_settings]

def subdir_name(tag, n):
    return f'{tag}_N{n}'


# -----------------------------------------------------------------------------
# W2 computation
# -----------------------------------------------------------------------------

def compute_w2_all_setups(base_dir, num_reps=100,
                          subsample=None, output_dir=None, seed=42):
    """Compute W2 distances to EP for all setups that have completed reps.

    Uses a two-track approach:
      - Grid-based W2 for exact/mean/eup (true distribution comparison)
      - Sample-based W2 for RKPCN (grid-sampled EP as reference)
    """
    base_dir = Path(base_dir)
    key = jr.key(seed)
    all_results = {}

    for tag, n, _ in setups:
        sname = subdir_name(tag, n)
        setup_dir = base_dir / sname

        if not setup_dir.exists():
            print(f'Skipping {sname} (directory not found)')
            continue

        completed, missing = check_completion_status(base_dir, sname, num_reps)
        if len(completed) == 0:
            print(f'Skipping {sname} (no completed reps)')
            continue

        key, subkey = jr.split(key)
        print(f'\nComputing W2 for {sname} ({len(completed)} reps)')

        results, eps = summarize_wasserstein_reps(
            key=subkey,
            base_dir=base_dir,
            subdir_name=sname,
            rep_idcs=completed,
            subsample=subsample,
            output_dir=output_dir,
        )
        all_results[sname] = results
        if eps is not None:
            print(f'  sample-based epsilon={eps:.6f}')

    return all_results


# -----------------------------------------------------------------------------
# Diagnostics summary
# -----------------------------------------------------------------------------

def print_diagnostics_all_setups(base_dir, num_reps=100):
    """Print acceptance rate summary for all setups."""
    base_dir = Path(base_dir)

    for tag, n, _ in setups:
        sname = subdir_name(tag, n)
        completed, _ = check_completion_status(base_dir, sname, num_reps)
        if len(completed) == 0:
            continue

        diag = summarize_diagnostics(base_dir, sname, completed)
        print(f'\n{sname} acceptance rates (median [min, max]):')
        for k in sorted(diag.keys()):
            vals = np.array(diag[k])
            print(f'  {k}: {np.median(vals):.4f} [{np.min(vals):.4f}, {np.max(vals):.4f}]')


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_w2_boxplots(w2_results, surrogate_tag, output_dir=None):
    """
    One box plot per design size, matching the PDE paper figure style.

    Methods are ordered left-to-right by descending median. Only plots
    'cut', 'rkpcn90', 'rkpcn95', 'rkpcn99', and 'eup' (no 'mean'/'exact').
    'rkpcn0' is relabeled 'cut'.

    Args:
        w2_results: dict mapping setup names to dicts of {method: (n_reps,) array}
        surrogate_tag: 'gp' or 'clip_gp'
        output_dir: if provided, save each figure as PDF here

    Returns:
        list of (fig, ax) tuples, one per design size
    """
    # Filter to this surrogate tag and sort by design size
    relevant = {}
    for k, v in w2_results.items():
        if k.startswith(surrogate_tag + '_N'):
            relevant[k] = v
    design_names = sorted(relevant.keys(), key=lambda s: int(s.split('_N')[1]))

    if not design_names:
        print(f"No results for {surrogate_tag}")
        return []

    figs = []
    for i, sname in enumerate(design_names):
        results = dict(relevant[sname])

        # Rename rkpcn0 → cut, drop mean/exact
        if 'rkpcn0' in results:
            results['cut'] = results.pop('rkpcn0')
        for drop in ['mean', 'exact']:
            results.pop(drop, None)

        if len(results) == 0:
            continue

        # Order by descending median
        method_order = sorted(results.keys(),
                              key=lambda m: np.median(results[m]),
                              reverse=True)
        data = [np.array(results[m]) for m in method_order]

        fig, ax = plt.subplots()
        ax.boxplot(data, tick_labels=method_order, showfliers=False)

        if i == 0:
            ax.set_ylabel('2-Wasserstein', fontsize=30)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=19)
        fig.tight_layout()

        if output_dir is not None:
            n = sname.split('_N')[1]
            out_path = Path(output_dir) / f'vsem_w2_{surrogate_tag}_N{n}.pdf'
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f'Saved: {out_path}')

        figs.append((fig, ax))

    return figs


def load_coverage_data(base_dir, subdir_name, rep_idcs):
    """Load and stack coverage data across replicates.

    Returns:
        dict with 'log_coverage' (n_reps, n_dists, n_probs),
        'probs' (n_probs,), 'dist_names' list
    """
    setup_dir = Path(base_dir) / subdir_name
    coverages = []

    for rep_idx in rep_idcs:
        cov_path = setup_dir / f'rep{rep_idx}' / 'coverage.npz'
        if cov_path.exists():
            cov = dict(jnp.load(cov_path))
            coverages.append(cov['log_coverage'])

    if len(coverages) == 0:
        return None

    first = dict(jnp.load(setup_dir / f'rep{rep_idcs[0]}' / 'coverage.npz'))
    return {
        'log_coverage': jnp.stack(coverages, axis=0),
        'probs': first['probs'],
        'dist_names': list(first['dist_names']),
    }


def plot_coverage_grid(base_dir, surrogate_tag, num_reps=100, figsize_scale=3.0):
    """Generate the 3x3 coverage plot for a surrogate type.

    Rows = design sizes (N=4, 8, 16), columns = approximations (mean, eup, ep).
    Each panel shows the median and 5th-95th percentile band of the coverage
    curve across replicates, matching the paper figure format.

    Args:
        base_dir: experiment base directory
        surrogate_tag: 'gp' or 'clip_gp'
        num_reps: total number of replicates
        figsize_scale: controls subplot size

    Returns:
        (fig, axes) or (None, None) if no data
    """
    from uncprop.utils.plot import set_plot_theme, plot_coverage_curve_reps

    colors = set_plot_theme()
    ndesign = [4, 8, 16]
    approx = ['mean', 'eup', 'ep']

    fig, axs = plt.subplots(len(ndesign), len(approx),
                            figsize=(figsize_scale * len(approx),
                                     figsize_scale * len(ndesign)))

    has_data = False

    for n_idx, n in enumerate(ndesign):
        sname = subdir_name(surrogate_tag, n)
        completed, _ = check_completion_status(base_dir, sname, num_reps)

        if len(completed) == 0:
            continue

        cov_data = load_coverage_data(base_dir, sname, completed)
        if cov_data is None:
            continue

        has_data = True
        dist_names = cov_data['dist_names']

        for dist_idx, dist_name in enumerate(approx):
            ax = axs[n_idx, dist_idx]

            if dist_name not in dist_names:
                continue

            idx = dist_names.index(dist_name)
            plot_coverage_curve_reps(
                log_coverage=cov_data['log_coverage'][:, [idx], :],
                probs=cov_data['probs'],
                names=[dist_name],
                colors=colors,
                qmin=0.05, qmax=0.95,
                single_plot=True,
                ax=ax,
                alpha=0.2,
            )

            # Format: titles on top row, y-labels on left column
            if n_idx == 0:
                ax.set_title(dist_name, fontsize=ax.title.get_fontsize() * 1.5)
            if n_idx != len(ndesign) - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('nominal')
            if dist_idx == 0:
                ax.set_ylabel(f'actual (N = {n})')
            else:
                ax.set_ylabel('')

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    if not has_data:
        plt.close(fig)
        return None, None

    fig.tight_layout()
    return fig, axs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='VSEM post-hoc analysis')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base output directory (default: ../../out/<experiment-name>)')
    parser.add_argument('--num-reps', type=int, default=100)
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample size for W2 computation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save W2 results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.base_dir is None:
        base_dir = repo_root / 'out' / args.experiment_name
    else:
        base_dir = Path(args.base_dir)

    print('=' * 60)
    print('VSEM Post-Hoc Analysis')
    print(f'  base_dir: {base_dir}')
    print(f'  num_reps: {args.num_reps}')
    print('=' * 60)

    # Diagnostics
    print('\n--- Diagnostics ---')
    print_diagnostics_all_setups(base_dir, args.num_reps)

    # Coverage plots
    print('\n--- Coverage Plots ---')
    for tag in gp_tags:
        fig, _ = plot_coverage_grid(base_dir, tag, num_reps=args.num_reps)
        if fig is not None:
            out_path = base_dir / f'vsem_coverage_{tag}.pdf'
            fig.savefig(out_path, bbox_inches='tight')
            print(f'Saved: {out_path}')

    # W2
    print('\n--- W2 Distances ---')
    w2 = compute_w2_all_setups(
        base_dir,
        num_reps=args.num_reps,
        subsample=args.subsample,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Print summary
    for sname, results in w2.items():
        print(f'\n{sname}:')
        for method, vals in sorted(results.items()):
            arr = np.array(vals)
            print(f'  {method}: median={np.median(arr):.4f}, mean={np.mean(arr):.4f}')

    # W2 box plots (one per design size, PDE paper style)
    print('\n--- W2 Box Plots ---')
    for tag in gp_tags:
        plot_w2_boxplots(w2, surrogate_tag=tag, output_dir=base_dir)


if __name__ == '__main__':
    main()
