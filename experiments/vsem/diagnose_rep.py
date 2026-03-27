"""
Diagnostic tool for investigating individual VSEM experiment replicates.

Generates trace plots, 2D KDE/contour comparisons, and autocorrelation
analysis. Not part of the main paper workflow — use for debugging and
understanding individual replicate behavior.

Usage:
    python diagnose_rep.py --experiment-name vsem_local_test --setup gp_N4 --rep 0
    python diagnose_rep.py --experiment-name vsem_scc_test --setup clip_gp_N4 --rep 3 --output-dir ./diag_out
"""
from jax import config
config.update('jax_enable_x64', True)

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rep_data(base_dir, setup_name, rep_idx):
    """Load all saved data for a single replicate."""
    rep_dir = Path(base_dir) / setup_name / f'rep{rep_idx}'

    data = {}
    data['samples'] = dict(jnp.load(rep_dir / 'samples.npz'))
    data['diagnostics'] = dict(jnp.load(rep_dir / 'diagnostics.npz'))
    data['coverage'] = dict(jnp.load(rep_dir / 'coverage.npz'))
    data['setup'] = dict(jnp.load(rep_dir / 'setup_info.npz'))

    gd_path = rep_dir / 'grid_densities.npz'
    if gd_path.exists():
        data['grid_densities'] = dict(jnp.load(gd_path))

    gi_path = rep_dir / 'grid_info.npz'
    if gi_path.exists():
        data['grid_info'] = dict(jnp.load(gi_path))

    return data


# ---------------------------------------------------------------------------
# Trace plots
# ---------------------------------------------------------------------------

def plot_trace(samples_dict, methods=None, param_idx=0, param_name=None,
               figsize=None):
    """Trace plots for a single parameter across methods.

    Args:
        samples_dict: {method_name: (n_samples, dim) array}
        methods: which methods to plot (default: all)
        param_idx: which parameter dimension to trace
        param_name: label for the parameter
    """
    if methods is None:
        methods = list(samples_dict.keys())
    n = len(methods)
    if figsize is None:
        figsize = (14, 2.5 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    label = param_name or f'param[{param_idx}]'

    for ax, method in zip(axes, methods):
        samp = np.array(samples_dict[method])
        ax.plot(samp[:, param_idx], lw=0.5, alpha=0.7)
        ax.set_ylabel(method, fontsize=12)
        ax.set_title(f'{method}: {label}', fontsize=11)

    axes[-1].set_xlabel('iteration')
    fig.tight_layout()
    return fig


def plot_trace_all_params(samples_dict, method, param_names=None,
                          figsize=None):
    """Trace plots for all parameters of a single method."""
    samp = np.array(samples_dict[method])
    n_samp, dim = samp.shape
    if param_names is None:
        param_names = [f'param[{i}]' for i in range(dim)]
    if figsize is None:
        figsize = (14, 2.5 * dim)

    fig, axes = plt.subplots(dim, 1, figsize=figsize, sharex=True)
    if dim == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(samp[:, i], lw=0.5, alpha=0.7)
        ax.set_ylabel(name, fontsize=12)

    axes[-1].set_xlabel('iteration')
    fig.suptitle(f'{method} trace', fontsize=14)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2D density contour comparison
# ---------------------------------------------------------------------------

def plot_2d_kde_comparison(samples_dict, methods=None, param_indices=(0, 1),
                           param_names=None, n_grid=100, levels=8,
                           figsize=None):
    """2D KDE contour overlay for comparing distributions.

    Args:
        samples_dict: {method_name: (n_samples, dim) array}
        methods: which methods to plot
        param_indices: tuple of two parameter dimensions to plot
        param_names: axis labels
        n_grid: grid resolution for KDE
        levels: number of contour levels
    """
    if methods is None:
        methods = list(samples_dict.keys())
    if param_names is None:
        param_names = [f'param[{i}]' for i in param_indices]
    if figsize is None:
        figsize = (8, 7)

    i, j = param_indices

    # Compute range from all methods
    all_x = np.concatenate([np.array(samples_dict[m])[:, i] for m in methods])
    all_y = np.concatenate([np.array(samples_dict[m])[:, j] for m in methods])
    pad = 0.1
    x_range = (all_x.min() - pad * np.ptp(all_x), all_x.max() + pad * np.ptp(all_x))
    y_range = (all_y.min() - pad * np.ptp(all_y), all_y.max() + pad * np.ptp(all_y))

    xg = np.linspace(*x_range, n_grid)
    yg = np.linspace(*y_range, n_grid)
    Xg, Yg = np.meshgrid(xg, yg)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])

    fig, ax = plt.subplots(figsize=figsize)
    cmap_names = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples',
                  'Greys', 'YlOrBr', 'PuBu']
    color_cycle = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange',
                   'tab:purple', 'tab:gray', 'tab:brown', 'tab:cyan']

    for idx, method in enumerate(methods):
        samp = np.array(samples_dict[method])
        x, y = samp[:, i], samp[:, j]

        try:
            kde = gaussian_kde(np.vstack([x, y]))
            Z = kde(positions).reshape(n_grid, n_grid)
            cmap = cmap_names[idx % len(cmap_names)]
            ax.contour(Xg, Yg, Z, levels=levels, cmap=cmap, alpha=0.7)
        except np.linalg.LinAlgError:
            # KDE fails if samples are degenerate
            ax.scatter(x, y, s=1, alpha=0.3, label=f'{method} (scatter)')

    # Legend
    handles = [Line2D([0], [0], color=color_cycle[idx % len(color_cycle)],
                       lw=2, label=m) for idx, m in enumerate(methods)]
    ax.legend(handles=handles, fontsize=11)

    ax.set_xlabel(param_names[0], fontsize=13)
    ax.set_ylabel(param_names[1], fontsize=13)
    fig.tight_layout()
    return fig


def plot_2d_kde_grid(samples_dict, reference='ep', methods=None,
                     param_indices=(0, 1), param_names=None,
                     n_grid=100, levels=8, figsize_per_panel=(5, 4.5)):
    """Side-by-side panels, each showing one method vs the reference.

    Useful for comparing each approximation to EP individually.
    """
    if methods is None:
        methods = [m for m in samples_dict.keys() if m != reference]
    if param_names is None:
        param_names = [f'param[{i}]' for i in param_indices]

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                             figsize_per_panel[1]))
    if n == 1:
        axes = [axes]

    i, j = param_indices

    # Shared range
    all_keys = methods + [reference]
    all_x = np.concatenate([np.array(samples_dict[m])[:, i] for m in all_keys])
    all_y = np.concatenate([np.array(samples_dict[m])[:, j] for m in all_keys])
    pad = 0.1
    x_range = (all_x.min() - pad * np.ptp(all_x), all_x.max() + pad * np.ptp(all_x))
    y_range = (all_y.min() - pad * np.ptp(all_y), all_y.max() + pad * np.ptp(all_y))
    xg = np.linspace(*x_range, n_grid)
    yg = np.linspace(*y_range, n_grid)
    Xg, Yg = np.meshgrid(xg, yg)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])

    # Reference KDE (may fail for degenerate samples)
    ref_samp = np.array(samples_dict[reference])
    try:
        ref_kde = gaussian_kde(np.vstack([ref_samp[:, i], ref_samp[:, j]]))
        Z_ref = ref_kde(positions).reshape(n_grid, n_grid)
        ref_ok = True
    except np.linalg.LinAlgError:
        ref_ok = False

    for ax, method in zip(axes, methods):
        samp = np.array(samples_dict[method])

        try:
            kde = gaussian_kde(np.vstack([samp[:, i], samp[:, j]]))
            Z = kde(positions).reshape(n_grid, n_grid)
            if ref_ok:
                ax.contour(Xg, Yg, Z_ref, levels=levels, cmap='Blues', alpha=0.6)
            else:
                ax.scatter(ref_samp[:, i], ref_samp[:, j], s=1, alpha=0.3, color='blue')
            ax.contour(Xg, Yg, Z, levels=levels, cmap='Reds', alpha=0.6)
        except np.linalg.LinAlgError:
            if ref_ok:
                ax.contour(Xg, Yg, Z_ref, levels=levels, cmap='Blues', alpha=0.6)
            else:
                ax.scatter(ref_samp[:, i], ref_samp[:, j], s=1, alpha=0.3, color='blue')
            ax.scatter(samp[:, i], samp[:, j], s=1, alpha=0.3, color='red')

        handles = [
            Line2D([0], [0], color='tab:blue', lw=2, label=reference),
            Line2D([0], [0], color='tab:red', lw=2, label=method),
        ]
        ax.legend(handles=handles, fontsize=10)
        ax.set_title(method, fontsize=13)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def plot_autocorrelation(samples_dict, methods=None, param_idx=0,
                         max_lag=200, param_name=None, figsize=(10, 5)):
    """Autocorrelation function for a single parameter across methods."""
    if methods is None:
        methods = list(samples_dict.keys())
    if param_name is None:
        param_name = f'param[{param_idx}]'

    fig, ax = plt.subplots(figsize=figsize)

    for method in methods:
        samp = np.array(samples_dict[method])[:, param_idx]
        samp = samp - samp.mean()
        n = len(samp)
        lags = min(max_lag, n - 1)

        # Normalized autocorrelation via FFT
        fft = np.fft.fft(samp, n=2 * n)
        acf = np.fft.ifft(fft * np.conj(fft)).real[:lags]
        acf /= acf[0]

        ax.plot(np.arange(lags), acf, label=method, lw=1.5)

    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('lag', fontsize=12)
    ax.set_ylabel('autocorrelation', fontsize=12)
    ax.set_title(f'Autocorrelation: {param_name}', fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def compute_ess(samples, method='batch_means', batch_size=None):
    """Estimate effective sample size for each parameter.

    Args:
        samples: (n_samples, dim) array
        method: 'batch_means' or 'autocorrelation'

    Returns:
        (dim,) array of ESS estimates
    """
    samples = np.array(samples)
    n, dim = samples.shape

    if method == 'batch_means':
        if batch_size is None:
            batch_size = max(1, int(np.sqrt(n)))
        n_batches = n // batch_size
        ess = np.zeros(dim)
        for j in range(dim):
            x = samples[:n_batches * batch_size, j]
            batch_means = x.reshape(n_batches, batch_size).mean(axis=1)
            var_bm = np.var(batch_means, ddof=1)
            var_x = np.var(x, ddof=1)
            if var_bm > 0:
                ess[j] = n * var_x / (batch_size * var_bm)
            else:
                ess[j] = n
        return ess

    elif method == 'autocorrelation':
        ess = np.zeros(dim)
        for j in range(dim):
            x = samples[:, j] - samples[:, j].mean()
            fft = np.fft.fft(x, n=2 * n)
            acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
            acf /= acf[0]

            # Initial monotone sequence estimator (Geyer 1992)
            tau = 1.0
            for k in range(1, n // 2):
                rho_pair = acf[2*k - 1] + acf[2*k]
                if rho_pair < 0:
                    break
                tau += 2 * rho_pair
            ess[j] = n / tau
        return ess

    else:
        raise ValueError(f'Unknown ESS method: {method}')


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(data, methods=None):
    """Print diagnostic summary for a replicate."""
    samples = data['samples']
    diagnostics = data['diagnostics']

    if methods is None:
        methods = list(samples.keys())

    print('\n=== Acceptance Rates ===')
    for k in sorted(diagnostics.keys()):
        print(f'  {k}: {float(diagnostics[k]):.4f}')

    print('\n=== Sample Shapes ===')
    for m in methods:
        print(f'  {m}: {np.array(samples[m]).shape}')

    print('\n=== Effective Sample Size (batch means) ===')
    for m in methods:
        samp = np.array(samples[m])
        ess = compute_ess(samp, method='batch_means')
        n = samp.shape[0]
        print(f'  {m}: ESS = {ess} (n={n}, ESS/n = {ess/n})')

    print('\n=== Sample Statistics ===')
    for m in methods:
        samp = np.array(samples[m])
        print(f'  {m}:')
        print(f'    mean = {samp.mean(axis=0)}')
        print(f'    std  = {samp.std(axis=0)}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Diagnose a single VSEM experiment replicate')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--setup', type=str, required=True,
                        help='e.g., gp_N4, clip_gp_N8')
    parser.add_argument('--rep', type=int, required=True)
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Save plots here (default: display only)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to include (default: all)')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.base_dir is None:
        base_dir = repo_root / 'out' / args.experiment_name
    else:
        base_dir = Path(args.base_dir)

    out_dir = None
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading rep {args.rep} from {args.setup}...')
    data = load_rep_data(base_dir, args.setup, args.rep)
    methods = args.methods or list(data['samples'].keys())

    # Summary
    print_summary(data, methods)

    # Param names from setup
    samp = data['samples']
    dim = np.array(samp[methods[0]]).shape[1]
    param_names = [f'u{i+1}' for i in range(dim)]

    def save_or_show(fig, name):
        if out_dir is not None:
            path = out_dir / f'{args.setup}_rep{args.rep}_{name}.pdf'
            fig.savefig(path, bbox_inches='tight')
            print(f'Saved: {path}')
        plt.close(fig)

    # Trace plots for each parameter
    for p_idx in range(dim):
        fig = plot_trace(samp, methods=methods, param_idx=p_idx,
                         param_name=param_names[p_idx])
        save_or_show(fig, f'trace_{param_names[p_idx]}')

    # 2D contour: all methods overlaid
    if dim >= 2:
        fig = plot_2d_kde_comparison(samp, methods=methods,
                                     param_indices=(0, 1),
                                     param_names=param_names[:2])
        save_or_show(fig, 'kde_overlay')

        # Side-by-side: each method vs EP
        compare_methods = [m for m in methods if m != 'ep']
        if 'ep' in methods and len(compare_methods) > 0:
            fig = plot_2d_kde_grid(samp, reference='ep',
                                   methods=compare_methods,
                                   param_indices=(0, 1),
                                   param_names=param_names[:2])
            save_or_show(fig, 'kde_vs_ep')

    # Autocorrelation
    for p_idx in range(dim):
        fig = plot_autocorrelation(samp, methods=methods, param_idx=p_idx,
                                   param_name=param_names[p_idx])
        save_or_show(fig, f'acf_{param_names[p_idx]}')

    if out_dir is None:
        plt.show()

    print('\nDone.')


if __name__ == '__main__':
    main()
