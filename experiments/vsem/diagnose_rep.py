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

def _log_dens_to_contour_data(log_dens_flat, grid_shape):
    """Convert flat log-density to normalized 2D density for contour plotting.

    Returns None if the density is degenerate (all -inf, all identical, etc.).
    """
    log_dens = np.array(log_dens_flat).ravel()

    # Filter out -inf and check for degeneracy
    finite_mask = np.isfinite(log_dens)
    if finite_mask.sum() < 10:
        return None

    # Shift for numerical stability and exponentiate
    log_max = log_dens[finite_mask].max()
    dens = np.exp(log_dens - log_max)
    dens[~finite_mask] = 0.0

    # Check for near-zero variance
    if dens.max() - dens.min() < 1e-15:
        return None

    return dens.reshape(grid_shape)


def _kde_contour_data(samples, param_indices, xg, yg):
    """Compute KDE on a grid from MCMC samples. Returns (Xg, Yg, Z) or None."""
    i, j = param_indices
    samp = np.array(samples)
    x, y = samp[:, i], samp[:, j]

    Xg, Yg = np.meshgrid(xg, yg)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])

    try:
        kde = gaussian_kde(np.vstack([x, y]))
        Z = kde(positions).reshape(Xg.shape)
        return Xg, Yg, Z
    except np.linalg.LinAlgError:
        return None


def _get_contour_for_method(method, grid_densities, samples_dict,
                            grid_info, param_indices):
    """Get 2D contour data for a method, preferring grid-based density.

    For methods with grid densities (exact, mean, eup, ep), uses the
    analytical density evaluated on the grid. For sample-only methods
    (rkpcn*), falls back to KDE from MCMC samples.

    Returns (Xg, Yg, Z) or None if neither approach works.
    """
    # Try grid-based density first
    if (grid_densities is not None and grid_info is not None
            and method in grid_densities):
        log_dens = grid_densities[method]
        shape = tuple(int(n) for n in grid_info['n_points_per_dim'])
        Z = _log_dens_to_contour_data(log_dens, shape)
        if Z is not None:
            low = np.array(grid_info['low'])
            high = np.array(grid_info['high'])
            xg = np.linspace(low[0], high[0], shape[0])
            yg = np.linspace(low[1], high[1], shape[1])
            Xg, Yg = np.meshgrid(xg, yg)
            return Xg, Yg, Z

    # Fall back to KDE from samples
    if method in samples_dict:
        samp = np.array(samples_dict[method])
        # Determine a reasonable grid range from samples
        i, j = param_indices
        x, y = samp[:, i], samp[:, j]
        if np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
            return None  # degenerate
        pad = 0.15
        xg = np.linspace(x.min() - pad * np.ptp(x),
                          x.max() + pad * np.ptp(x), 80)
        yg = np.linspace(y.min() - pad * np.ptp(y),
                          y.max() + pad * np.ptp(y), 80)
        return _kde_contour_data(samp, param_indices, xg, yg)

    return None


def plot_contour_vs_ep(data, methods=None, param_names=None,
                       levels=10, figsize_per_panel=(5, 4.5),
                       ncols=3):
    """Side-by-side panels, each showing one method (red) vs EP (blue).

    Uses grid-based densities when available; falls back to KDE for
    sample-only methods (rkpcn).

    Args:
        data: dict from load_rep_data (must have 'samples', may have
              'grid_densities' and 'grid_info')
        methods: list of method names to compare against EP
        param_names: axis labels [u1, u2]
        levels: number of contour levels
    """
    samples = data['samples']
    grid_densities = data.get('grid_densities')
    grid_info = data.get('grid_info')
    param_indices = (0, 1)

    if param_names is None:
        dim = np.array(samples[list(samples.keys())[0]]).shape[1]
        param_names = [f'u{i+1}' for i in range(dim)]

    if methods is None:
        methods = [m for m in samples.keys() if m != 'ep']

    # EP reference contour
    ep_data = _get_contour_for_method('ep', grid_densities, samples,
                                       grid_info, param_indices)

    n = len(methods)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0] * ncols,
                                       figsize_per_panel[1] * nrows),
                              squeeze=False)

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols, idx % ncols]

        method_data = _get_contour_for_method(method, grid_densities,
                                               samples, grid_info,
                                               param_indices)

        # Plot EP reference
        if ep_data is not None:
            Xg_ep, Yg_ep, Z_ep = ep_data
            ax.contour(Xg_ep, Yg_ep, Z_ep, levels=levels,
                       cmap='Blues', alpha=0.6)

        # Plot this method
        if method_data is not None:
            Xg_m, Yg_m, Z_m = method_data
            ax.contour(Xg_m, Yg_m, Z_m, levels=levels,
                       cmap='Reds', alpha=0.6)
        elif method in samples:
            # Last resort: scatter
            samp = np.array(samples[method])
            ax.scatter(samp[:, 0], samp[:, 1], s=1, alpha=0.3, color='red')

        handles = [
            Line2D([0], [0], color='tab:blue', lw=2, label='ep'),
            Line2D([0], [0], color='tab:red', lw=2, label=method),
        ]
        ax.legend(handles=handles, fontsize=10)
        ax.set_title(method, fontsize=13)
        ax.set_xlabel(param_names[0])
        ax.set_ylabel(param_names[1])

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

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

    # 2D contour: each method vs EP (separate panels)
    if dim >= 2:
        compare_methods = [m for m in methods if m != 'ep']
        if 'ep' in methods and len(compare_methods) > 0:
            fig = plot_contour_vs_ep(data, methods=compare_methods,
                                     param_names=param_names[:2])
            save_or_show(fig, 'contour_vs_ep')

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
