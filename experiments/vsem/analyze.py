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

def plot_w2_boxplots(w2_results, title=None, figsize=(10, 5)):
    """
    Box plots of W2 distance to EP, grouped by approximation method.

    Args:
        w2_results: dict mapping setup names to dicts of {method: (n_reps,) array}
        title: optional plot title
        figsize: figure size

    Returns:
        (fig, axes) — one subplot per setup
    """
    setup_names = list(w2_results.keys())
    n_setups = len(setup_names)

    if n_setups == 0:
        print("No results to plot")
        return None, None

    fig, axes = plt.subplots(1, n_setups, figsize=(figsize[0], figsize[1]),
                             sharey=True, squeeze=False)
    axes = axes.ravel()

    for i, sname in enumerate(setup_names):
        ax = axes[i]
        results = w2_results[sname]

        if len(results) == 0:
            ax.set_title(sname)
            continue

        method_names = sorted(results.keys())
        data = [np.array(results[m]) for m in method_names]

        bp = ax.boxplot(data, tick_labels=method_names, vert=True, patch_artist=True)
        ax.set_title(sname)
        ax.set_ylabel('W₂ to EP' if i == 0 else '')
        ax.tick_params(axis='x', rotation=45)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_w2_by_design_size(w2_results, surrogate_tag='gp', figsize=(8, 5)):
    """
    Box plots comparing W2 across design sizes for a single surrogate type.

    Args:
        w2_results: dict from compute_w2_all_setups
        surrogate_tag: 'gp' or 'clip_gp'
        figsize: figure size

    Returns:
        (fig, axes)
    """
    # Filter to relevant setups
    relevant = {k: v for k, v in w2_results.items() if k.startswith(surrogate_tag)}
    if not relevant:
        print(f"No results for {surrogate_tag}")
        return None, None

    # Get all method names across setups
    all_methods = set()
    for v in relevant.values():
        all_methods.update(v.keys())
    methods = sorted(all_methods)

    if len(methods) == 0:
        print(f"No methods with results for {surrogate_tag}")
        return None, None

    n_designs = len(relevant)
    design_names = sorted(relevant.keys())

    fig, axes = plt.subplots(1, len(methods), figsize=figsize, sharey=True, squeeze=False)
    axes = axes.ravel()

    for j, method in enumerate(methods):
        ax = axes[j]
        data = []
        labels = []
        for dname in design_names:
            if method in relevant[dname]:
                data.append(np.array(relevant[dname][method]))
                # Extract N from name like 'gp_N4'
                labels.append(dname.split('_N')[1])

        if data:
            ax.boxplot(data, tick_labels=labels, vert=True, patch_artist=True)

        ax.set_title(method)
        ax.set_xlabel('N (design size)')
        ax.set_ylabel('W₂ to EP' if j == 0 else '')

    fig.suptitle(f'{surrogate_tag}: W₂ to EP by design size')
    fig.tight_layout()

    return fig, axes


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

    # Plots
    fig, _ = plot_w2_boxplots(w2, title='W₂ to EP across setups')
    if fig is not None:
        out_path = base_dir / 'w2_boxplots.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        print(f'\nSaved: {out_path}')

    for tag in gp_tags:
        fig, _ = plot_w2_by_design_size(w2, surrogate_tag=tag)
        if fig is not None:
            out_path = base_dir / f'w2_by_design_{tag}.pdf'
            fig.savefig(out_path, bbox_inches='tight')
            print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
