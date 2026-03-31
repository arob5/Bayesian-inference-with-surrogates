# experiments/vsem/rkpcn_analysis/diagnostics.py
"""
Diagnostic summaries and tables for RKPCN analysis.

Provides functions to:
  - Compute and display summary statistics tables
  - Compute autocorrelation functions
  - Compare W2 distances to a grid-based EP baseline
"""

import jax.numpy as jnp
import numpy as np

from uncprop.utils.diagnostics import compute_ess
from uncprop.utils.wasserstein import wasserstein2_sinkhorn


def autocorrelation(x, max_lag=2000):
    """Compute autocorrelation of a 1D array up to max_lag.

    Args:
        x: 1D array.
        max_lag: Maximum lag to compute.

    Returns:
        1D array of length max_lag.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode='full')
    acf = acf[n - 1:n - 1 + max_lag] / (var * n)
    return acf


def integrated_autocorrelation_time(x, max_lag=2000):
    """Estimate integrated autocorrelation time (IAT) for a 1D array.

    Uses the initial positive sequence estimator: sums autocorrelation
    pairs until the first negative pair sum.

    Args:
        x: 1D array.
        max_lag: Maximum lag for ACF computation.

    Returns:
        Estimated IAT (float).
    """
    acf = autocorrelation(x, max_lag=max_lag)
    # Sum consecutive pairs until first negative pair
    iat = acf[0]  # = 1.0
    for k in range(1, max_lag - 1, 2):
        pair_sum = acf[k] + acf[k + 1] if k + 1 < max_lag else acf[k]
        if pair_sum < 0:
            break
        iat += 2 * pair_sum
    return max(1.0, iat)


def summary_table(results, par_names=None):
    """Print a summary table of RKPCN variant results.

    Args:
        results: dict mapping label -> result dict (from run_rkpcn_variant).
        par_names: parameter names for ESS columns.

    Returns:
        dict of dicts (label -> {metric: value}) for programmatic access.
    """
    if not results:
        print('No results to summarize.')
        return {}

    first = next(iter(results.values()))
    d = first['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{i+1}' for i in range(d)]

    # Header
    ess_cols = ''.join(f' {f"ESS({p})":>10s}' for p in par_names)
    header = (f'{"label":>16s} | {"rho":>5s} | {"accept":>7s} | '
              f'{"min ESS":>8s} |{ess_cols} | '
              f'{"n_post":>7s} | {"IAT(ld)":>8s}')
    print(header)
    print('-' * len(header))

    table = {}
    for label, res in results.items():
        pos = res['post_burnin']
        ess = compute_ess(pos)
        ld = res['logdensities'][res['n_burnin']:]
        iat_ld = integrated_autocorrelation_time(ld)

        ess_str = ''.join(f' {e:10.1f}' for e in ess)
        print(f'{label:>16s} | {res["rho"]:5.2f} | '
              f'{res["accept_rate"]:7.4f} | '
              f'{min(ess):8.1f} |{ess_str} | '
              f'{pos.shape[0]:7d} | {iat_ld:8.1f}')

        table[label] = {
            'rho': res['rho'],
            'accept_rate': res['accept_rate'],
            'min_ess': min(ess),
            'ess': ess,
            'n_post': pos.shape[0],
            'iat_logdensity': iat_ld,
        }

    return table


def w2_table(results, ep_grid_density, grid, par_names=None,
             thin=5, sinkhorn_kwargs=None):
    """Compute and print W2 distances from each variant to the grid-based EP.

    Uses KDE-on-grid approach: fits a KDE to the RKPCN samples, evaluates
    on the grid, normalizes, and computes grid-based W2 against the EP density.

    Also includes exact, mean, eup samples if provided in results.

    Args:
        results: dict mapping label -> result dict (must have 'post_burnin').
        ep_grid_density: (n_grid,) array of (unnormalized) log EP density on grid.
        grid: Grid object.
        par_names: Parameter names for display.
        thin: Thinning factor for samples before KDE.
        sinkhorn_kwargs: kwargs for Sinkhorn solver.

    Returns:
        dict mapping label -> W2 distance.
    """
    from uncprop.utils.grid import normalize_density_over_grid
    from scipy.stats import gaussian_kde

    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {'threshold': 1e-6, 'max_iterations': 5000}

    # Normalize EP density on grid
    logp_ep_norm = normalize_density_over_grid(
        ep_grid_density, cell_area=grid.cell_area)[0].squeeze()

    grid_pts = np.array(grid.flat_grid)

    print(f'{"method":>16s} | {"W2 to EP":>10s} | {"n_samp":>8s}')
    print('-' * 45)

    w2_results = {}

    for label, res in results.items():
        samp = res['post_burnin'][::thin]
        n_samp = samp.shape[0]

        if n_samp < 10:
            print(f'{label:>16s} | {"(too few)":>10s} | {n_samp:8d}')
            continue

        try:
            # KDE on grid
            kde = gaussian_kde(samp.T)
            log_kde = np.log(kde(grid_pts.T) + 1e-300)

            logp_kde_norm = normalize_density_over_grid(
                jnp.array(log_kde), cell_area=grid.cell_area)[0].squeeze()

            # Grid-based W2 via Sinkhorn
            from ott.geometry import grid as ott_grid
            from ott.problems.linear import linear_problem
            from ott.solvers.linear import sinkhorn

            x = [jnp.array(grid.axes[i]) for i in range(grid.n_dims)]
            geom = ott_grid.Grid(x=x, epsilon=0.01)

            a = jnp.exp(logp_ep_norm)
            b = jnp.exp(logp_kde_norm)

            # Ensure valid probability vectors
            a = jnp.clip(a, 0.0)
            b = jnp.clip(b, 0.0)
            a = a / a.sum()
            b = b / b.sum()

            prob = linear_problem.LinearProblem(geom, a=a, b=b)
            solver = sinkhorn.Sinkhorn(**sinkhorn_kwargs)
            out = solver(prob)
            w2 = float(jnp.sqrt(jnp.clip(out.reg_ot_cost, 0.0)))

            print(f'{label:>16s} | {w2:10.4f} | {n_samp:8d}')
            w2_results[label] = w2

        except Exception as e:
            print(f'{label:>16s} | {"FAILED":>10s} | {n_samp:8d}  ({e})')

    return w2_results
