"""
Post-hoc analysis for PDE experiment results.

Computes W2 distances, summarizes diagnostics, computes coverage, and
generates plots. Run after the experiment has completed (all reps saved
to disk).

Usage:
    # From experiments/elliptic_pde/ directory:
    python analyze.py --experiment-name pde_experiment

    # Or import functions in a notebook:
    from analyze import compute_w2_all_designs, plot_w2_boxplots
"""
from jax import config
config.update('jax_enable_x64', True)

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular
import numpy as np
import matplotlib.pyplot as plt

from uncprop.utils.wasserstein import compute_wasserstein_comparison
from uncprop.custom_types import PRNGKey, Array


# ---- Setup definitions ----
# Default design sizes for the full experiment (matches runner.py).
# discover_design_sizes() can find whatever is on disk.
DEFAULT_DESIGN_SIZES = [10, 20, 30]


def subdir_name(n_design):
    return f'n_design_{n_design}'


def discover_design_sizes(base_dir):
    """Scan base_dir for n_design_* subdirectories and return sorted list."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []
    sizes = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith('n_design_'):
            try:
                sizes.append(int(p.name.replace('n_design_', '')))
            except ValueError:
                pass
    return sorted(sizes)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def check_completion_status(base_dir, n_design, num_reps):
    """Return (completed_rep_idcs, missing_rep_idcs)."""
    setup_dir = Path(base_dir) / subdir_name(n_design)
    if not setup_dir.exists():
        return [], list(range(num_reps))

    completed = []
    missing = []
    for i in range(num_reps):
        rep_dir = setup_dir / f'rep{i}'
        if (rep_dir / 'samples.npz').exists() and (rep_dir / 'diagnostics.npz').exists():
            completed.append(i)
        else:
            missing.append(i)

    return completed, missing


def read_samp(base_dir, n_design, rep_idx):
    rep_dir = Path(base_dir) / subdir_name(n_design) / f'rep{rep_idx}'
    samp = dict(jnp.load(rep_dir / 'samples.npz'))

    # Legacy: also load rkpcn_samples.npz if it exists
    rkpcn_path = rep_dir / 'rkpcn_samples.npz'
    if rkpcn_path.exists():
        rkpcn_samp = dict(jnp.load(rkpcn_path))
        samp = samp | rkpcn_samp

    return samp


def read_diagnostics(base_dir, n_design, rep_idx):
    rep_dir = Path(base_dir) / subdir_name(n_design) / f'rep{rep_idx}'
    return dict(jnp.load(rep_dir / 'diagnostics.npz'))


# -----------------------------------------------------------------------------
# Diagnostics summary
# -----------------------------------------------------------------------------

def summarize_diagnostics(base_dir, n_design, rep_idcs):
    """Aggregate diagnostics across replicates."""
    all_diag = {}
    for rep_idx in rep_idcs:
        try:
            diag = read_diagnostics(base_dir, n_design, rep_idx)
            for k, v in diag.items():
                all_diag.setdefault(k, []).append(float(v))
        except Exception as e:
            print(f'  Warning: failed to load diagnostics for rep {rep_idx}: {e}')
    return all_diag


def print_diagnostics_all_designs(base_dir, num_reps=100, design_sizes=None):
    """Print acceptance rate summary for all design sizes."""
    base_dir = Path(base_dir)
    if design_sizes is None:
        design_sizes = discover_design_sizes(base_dir)

    for n in design_sizes:
        completed, _ = check_completion_status(base_dir, n, num_reps)
        if len(completed) == 0:
            continue

        diag = summarize_diagnostics(base_dir, n, completed)
        print(f'\nn_design = {n} acceptance rates (median [min, max]):')
        for k in sorted(diag.keys()):
            if not k.endswith('_accept_rate'):
                continue
            vals = np.array(diag[k])
            print(f'  {k}: {np.median(vals):.4f} [{np.min(vals):.4f}, {np.max(vals):.4f}]')


# -----------------------------------------------------------------------------
# Coverage
# -----------------------------------------------------------------------------

def estimate_mahalanobis_coverage(
    samples: dict[str, Array],
    baseline: str,
    probs: Array,
    jitter: float = 1e-8
) -> dict[str, Array]:
    """
    Estimate joint (ellipsoidal) coverage using Mahalanobis distance.

    For each approximating distribution X and each probability level p,
    coverage is the fraction of baseline samples falling inside the
    p-level ellipsoid of X.

    Args:
        samples: dict mapping names to (n, d) sample arrays
        baseline: key for the baseline (ground-truth) distribution
        probs: (m,) array of probability levels in (0, 1]

    Returns:
        dict mapping names to (m,) coverage arrays (baseline excluded)
    """
    baseline_samples = samples[baseline]
    d = baseline_samples.shape[1]

    def mahalanobis_sq(x, mean, chol):
        diff = x - mean
        y = solve_triangular(chol, diff.T, lower=True).T
        return jnp.sum(y**2, axis=-1)

    results = {}

    for name, x in samples.items():
        if name == baseline:
            continue

        mean_x = jnp.mean(x, axis=0)
        cov_x = jnp.cov(x, rowvar=False) + jitter * jnp.eye(d)
        chol_x = jnp.linalg.cholesky(cov_x, upper=False)

        r2_x = mahalanobis_sq(x, mean_x, chol_x)
        r2_baseline = mahalanobis_sq(baseline_samples, mean_x, chol_x)

        thresholds = jnp.quantile(r2_x, probs)
        coverage = jnp.mean(r2_baseline[:, None] <= thresholds[None, :], axis=0)

        results[name] = coverage

    return results


def assemble_coverage_reps(base_dir, n_design, probs,
                           approx_dist_names, baseline='exact',
                           num_reps=100):
    """Returns (n_reps, n_approx_dists, n_probs) array of coverage values."""
    base_dir = Path(base_dir)
    dist_order = [baseline] + approx_dist_names
    coverage_list = []

    completed, _ = check_completion_status(base_dir, n_design, num_reps)

    for rep_idx in completed:
        samples = read_samp(base_dir, n_design, rep_idx)
        # Only include distributions that exist in this rep's samples
        available = {nm: samples[nm] for nm in dist_order if nm in samples}
        if baseline not in available:
            continue

        coverage = estimate_mahalanobis_coverage(
            samples=available, baseline=baseline, probs=probs)
        coverage_list.append(jnp.stack(list(coverage.values())))

    if len(coverage_list) == 0:
        return None

    return jnp.stack(coverage_list)


def plot_coverage_grid(base_dir, num_reps=100, design_sizes=None,
                       figsize_scale=3.0):
    """Generate coverage plot: rows = design sizes, cols = approximations."""
    base_dir = Path(base_dir)
    if design_sizes is None:
        design_sizes = discover_design_sizes(base_dir)
    if len(design_sizes) == 0:
        return None, None

    probs = jnp.linspace(0.05, 0.99, 20)
    approx_names = ['mean', 'eup', 'ep_mcwmh']
    display_names = {'mean': 'mean', 'eup': 'eup', 'ep_mcwmh': 'ep'}

    fig, axs = plt.subplots(len(design_sizes), len(approx_names),
                            figsize=(figsize_scale * len(approx_names),
                                     figsize_scale * len(design_sizes)),
                            squeeze=False)

    has_data = False

    for n_idx, n in enumerate(design_sizes):
        completed, _ = check_completion_status(base_dir, n, num_reps)
        if len(completed) == 0:
            continue

        coverage = assemble_coverage_reps(
            base_dir, n, probs, approx_names,
            baseline='exact', num_reps=num_reps)

        if coverage is None:
            continue

        has_data = True

        for dist_idx, dist_name in enumerate(approx_names):
            ax = axs[n_idx, dist_idx]

            # coverage shape: (n_reps, n_approx, n_probs)
            cov_vals = np.array(coverage[:, dist_idx, :])
            median = np.median(cov_vals, axis=0)
            q05 = np.quantile(cov_vals, 0.05, axis=0)
            q95 = np.quantile(cov_vals, 0.95, axis=0)

            ax.fill_between(probs, q05, q95, alpha=0.2, color='blue')
            ax.plot(probs, median, color='blue')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            if n_idx == 0:
                ax.set_title(display_names[dist_name],
                             fontsize=ax.title.get_fontsize() * 1.5)
            if n_idx == len(design_sizes) - 1:
                ax.set_xlabel('nominal')
            else:
                ax.set_xlabel('')
            if dist_idx == 0:
                ax.set_ylabel(f'actual (N = {n})')
            else:
                ax.set_ylabel('')

    if not has_data:
        plt.close(fig)
        return None, None

    fig.tight_layout()
    return fig, axs


# -----------------------------------------------------------------------------
# W2 computation
# -----------------------------------------------------------------------------

def summarize_wasserstein_design_reps(key, base_dir, n_design, rep_idcs,
                                      subsample=1000, output_dir=None):
    """
    Compute W2 distance to EP (MCwMH) for all reps for a given design size.

    Uses consistent regularization epsilon across all reps.
    """
    base_dir = Path(base_dir)
    w2_keys = jr.split(key, len(rep_idcs))
    results = []
    eps = None

    if output_dir is not None:
        output_path = Path(output_dir) / f'w2_ndesign_{n_design}.npz'
    else:
        output_path = None

    def _combine_results(res):
        keys = res[0].keys()
        return {k: jnp.stack([r[k] for r in res]) for k in keys}

    for i, rep_idx in enumerate(rep_idcs):
        try:
            print(f'  Rep {rep_idx}')
            samp = read_samp(base_dir, n_design, rep_idx)

            rep_results, eps = compute_wasserstein_comparison(
                samples=samp,
                reference_key='ep_mcwmh',
                subsample=subsample,
                key=w2_keys[i],
                epsilon=eps,
                sinkhorn_kwargs={'threshold': 1e-6, 'max_iterations': 5000,
                                 'lse_mode': True}
            )
            results.append(rep_results)

            if i > 0 and i % 20 == 0:
                if output_path is not None:
                    jnp.savez(output_path, **_combine_results(results))
                jax.clear_caches()
        except Exception as e:
            print(f'  Failed rep {rep_idx}: {e}')

    if len(results) == 0:
        return {}, eps

    results = _combine_results(results)

    if output_path is not None:
        jnp.savez(output_path, **results)

    return results, eps


def compute_w2_all_designs(base_dir, num_reps=100, subsample=1000,
                           output_dir=None, seed=42, design_sizes=None):
    """Compute W2 distances to EP for all design sizes."""
    base_dir = Path(base_dir)
    if design_sizes is None:
        design_sizes = discover_design_sizes(base_dir)
    key = jr.key(seed)
    all_results = {}

    for n in design_sizes:
        completed, missing = check_completion_status(base_dir, n, num_reps)
        if len(completed) == 0:
            print(f'Skipping n_design={n} (no completed reps)')
            continue

        key, subkey = jr.split(key)
        print(f'\nComputing W2 for n_design={n} ({len(completed)} reps)')

        results, eps = summarize_wasserstein_design_reps(
            key=subkey, base_dir=base_dir, n_design=n,
            rep_idcs=completed, subsample=subsample,
            output_dir=output_dir)
        all_results[n] = results
        if eps is not None:
            print(f'  epsilon={eps:.6f}')

    return all_results


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_w2_boxplots(w2_results, output_dir=None):
    """
    One box plot per design size, matching VSEM/paper style.

    Methods ordered left-to-right by descending median. No fill,
    horizontal x-axis labels, 'cut' for rkpcn0, no title.
    Only plots RKPCN and EUP methods (no 'mean', 'exact').
    """
    figs = []

    for i, n in enumerate(sorted(w2_results.keys())):
        results = dict(w2_results[n])

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
            out_path = Path(output_dir) / f'pde_w2_ndesign_{n}.pdf'
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f'Saved: {out_path}')

        figs.append((fig, ax))

    return figs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PDE post-hoc analysis')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base output directory (default: ../../out/<experiment-name>)')
    parser.add_argument('--num-reps', type=int, default=100)
    parser.add_argument('--subsample', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.base_dir is None:
        base_dir = repo_root / 'out' / args.experiment_name
    else:
        base_dir = Path(args.base_dir)

    print('=' * 60)
    print('PDE Post-Hoc Analysis')
    print(f'  base_dir: {base_dir}')
    print(f'  num_reps: {args.num_reps}')
    print('=' * 60)

    # Discover what's on disk
    design_sizes = discover_design_sizes(base_dir)
    print(f'  design sizes found: {design_sizes}')

    # Status
    print('\n--- Status ---')
    for n in design_sizes:
        completed, missing = check_completion_status(base_dir, n, args.num_reps)
        print(f'  n_design={n}: {len(completed)} completed, {len(missing)} missing')

    # Diagnostics
    print('\n--- Diagnostics ---')
    print_diagnostics_all_designs(base_dir, args.num_reps)

    # Coverage plots
    print('\n--- Coverage Plots ---')
    fig, _ = plot_coverage_grid(base_dir, num_reps=args.num_reps)
    if fig is not None:
        out_path = base_dir / 'pde_coverage.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved: {out_path}')

    # W2
    print('\n--- W2 Distances ---')
    w2 = compute_w2_all_designs(
        base_dir, num_reps=args.num_reps,
        subsample=args.subsample, seed=args.seed,
        output_dir=base_dir)

    # Print summary
    for n, results in sorted(w2.items()):
        print(f'\nn_design={n}:')
        for method, vals in sorted(results.items()):
            arr = np.array(vals)
            print(f'  {method}: median={np.median(arr):.4f}, mean={np.mean(arr):.4f}')

    # W2 box plots
    print('\n--- W2 Box Plots ---')
    plot_w2_boxplots(w2, output_dir=base_dir)


if __name__ == '__main__':
    main()
