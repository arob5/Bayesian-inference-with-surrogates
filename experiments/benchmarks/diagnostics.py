# experiments/benchmarks/diagnostics.py
"""
Diagnostic summaries for benchmark variants.

Provides:
  - ACF + integrated autocorrelation time (IAT)
  - Summary tables (ESS, accept, IAT, runtime)
  - VSEM-specific W2-to-EP comparison (only valid when a 2D grid-based
    EP density is available; gated by caller).

All functions here are surrogate-agnostic except ``w2_vs_grid_ep``,
which requires a grid and is applied only to VSEM replicates.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from uncprop.utils.diagnostics import compute_ess


# =============================================================================
# Autocorrelation
# =============================================================================

def autocorrelation(x, max_lag: int = 2000):
    """ACF of a 1D series up to ``max_lag``. Zero-returning on degenerate input."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 2:
        return np.zeros(max_lag)
    x = x - np.mean(x)
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag)
    acf = np.correlate(x, x, mode='full')
    return acf[n - 1: n - 1 + max_lag] / (var * n)


def integrated_autocorrelation_time(x, max_lag: int = 2000) -> float:
    """Initial-positive-sequence estimator of IAT.

    Sums consecutive pairs of ACF lags until the first negative pair sum.
    """
    acf = autocorrelation(x, max_lag=max_lag)
    iat = float(acf[0])
    for k in range(1, max_lag - 1, 2):
        pair = acf[k] + acf[k + 1] if k + 1 < max_lag else acf[k]
        if pair < 0:
            break
        iat += 2 * float(pair)
    return max(1.0, iat)


# =============================================================================
# Per-variant summary table
# =============================================================================

def summary_table(results: dict, par_names: list[str] | None = None) -> dict:
    """Print and return a per-variant summary of benchmark results.

    Parameters
    ----------
    results : dict
        Label -> result dict (either from a fresh benchmark run or from
        ``load_benchmark_results``). Must have ``post_burnin`` and
        ``logdensities``; rows with no post-burnin samples are skipped.
    par_names : list[str] or None
    """
    if not results:
        print('No results to summarize.')
        return {}

    first = next(iter(results.values()))
    post = first.get('post_burnin')
    if post is None:
        print('No post-burnin samples in results.')
        return {}
    d = post.shape[1]
    if par_names is None:
        par_names = [f'u{i}' for i in range(d)]

    ess_cols = ''.join(f' {f"ESS({p})":>10s}' for p in par_names)
    header = (f'{"label":>20s} | {"rho":>5s} | {"accept":>7s} | '
              f'{"min ESS":>8s} |{ess_cols} | '
              f'{"n_post":>7s} | {"IAT(ld)":>8s}')
    print(header)
    print('-' * len(header))

    table = {}
    for label, res in results.items():
        pos = res.get('post_burnin')
        if pos is None:
            continue
        ess = res.get('ess', compute_ess(pos))
        ld = res.get('logdensities', np.empty(0))
        n_burnin = res.get('n_burnin', 0)
        ld_post = ld[n_burnin:] if ld.size > 0 else np.empty(0)

        # Prefer IAT from the saved summary (computed at run time on
        # the full trace). Fall back to recomputation from post-burnin
        # log-densities — only available when full traces were saved.
        saved_iat = res.get('summary', {}).get('iat_logdensity')
        if saved_iat is not None and not (isinstance(saved_iat, float)
                                           and np.isnan(saved_iat)):
            iat_ld = float(saved_iat)
        elif ld_post.size > 1:
            iat_ld = integrated_autocorrelation_time(ld_post)
        else:
            iat_ld = float('nan')

        ess_str = ''.join(f' {e:10.1f}' for e in ess)
        print(f'{label:>20s} | {res.get("rho", 0):5.2f} | '
              f'{res.get("accept_rate", 0):7.4f} | '
              f'{min(ess):8.1f} |{ess_str} | '
              f'{pos.shape[0]:7d} | {iat_ld:8.1f}')

        table[label] = {
            'rho': res.get('rho'),
            'accept_rate': res.get('accept_rate'),
            'min_ess': float(min(ess)),
            'ess': [float(e) for e in ess],
            'n_post': int(pos.shape[0]),
            'iat_logdensity': float(iat_ld),
        }

    return table


# =============================================================================
# W2 vs. grid-based EP (VSEM only)
# =============================================================================

def w2_vs_grid_ep(
    results: dict,
    ep_grid_density: np.ndarray,
    grid,
    thin: int = 5,
    sinkhorn_kwargs: dict | None = None,
):
    """Compute W2 between each variant's sample KDE and a grid-based EP density.

    VSEM-specific: requires a 2D grid over which the EP density is
    defined. For experiments without such a grid (e.g. elliptic PDE),
    the caller should skip this function.

    Parameters
    ----------
    results : dict
        Label -> result dict. Expected: ``post_burnin`` (required),
        ``sample_weights`` (optional; used to weight the KDE).
    ep_grid_density : (n_grid,) array
        (Unnormalized) log EP density on the grid points.
    grid : :class:`uncprop.utils.grid.Grid`
    thin : int
        Thinning applied to samples before fitting the KDE.
    sinkhorn_kwargs : dict or None
        Passed to ``ott.solvers.linear.sinkhorn.Sinkhorn``.

    Returns
    -------
    dict : label -> W2 value (or missing if the variant failed).
    """
    import warnings
    from scipy.stats import gaussian_kde
    from ott.geometry import pointcloud as ott_pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    from uncprop.utils.grid import normalize_density_over_grid

    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {'threshold': 1e-6, 'max_iterations': 5000}

    logp_ep_norm = normalize_density_over_grid(
        ep_grid_density, cell_area=grid.cell_area)[0].squeeze()

    grid_pts = np.array(grid.flat_grid)
    scale = np.array(grid.high - grid.low)
    scale = np.where(scale > 0, scale, 1.0)
    pts_norm = jnp.array((grid_pts - np.array(grid.low)) / scale)

    min_prob = 1e-30
    a = jnp.exp(logp_ep_norm)
    a = jnp.where(a < min_prob, 0.0, a)
    a = a / a.sum()

    print(f'{"method":>20s} | {"W2 to EP":>10s} | {"converged":>10s} | '
          f'{"n_samp":>8s}')
    print('-' * 60)

    w2_results = {}
    for label, res in results.items():
        samp = res.get('post_burnin')
        if samp is None or samp.shape[0] < 10:
            n = 0 if samp is None else samp.shape[0]
            print(f'{label:>20s} | {"(too few)":>10s} | {"":>10s} | {n:8d}')
            continue
        samp = samp[::thin]
        n_samp = samp.shape[0]

        try:
            sw = res.get('sample_weights')
            if sw is not None:
                sw = np.asarray(sw)[::thin]
                tot = sw.sum()
                sw = sw / tot if tot > 0 else None

            kde = gaussian_kde(samp.T, weights=sw)
            log_kde = np.log(np.maximum(kde(grid_pts.T), 1e-300))

            logp_kde_norm = normalize_density_over_grid(
                jnp.array(log_kde), cell_area=grid.cell_area)[0].squeeze()

            b = jnp.exp(logp_kde_norm)
            b = jnp.where(b < min_prob, 0.0, b)
            b = b / b.sum()

            n_nonzero_b = int(jnp.sum(b > 0))
            if n_nonzero_b < 5:
                print(f'{label:>20s} | {"DEGEN":>10s} | {"":>10s} | '
                      f'{n_samp:8d}  (KDE: {n_nonzero_b} nonzero cells)')
                continue

            geom = ott_pointcloud.PointCloud(pts_norm, pts_norm, epsilon=None)
            prob = linear_problem.LinearProblem(geom, a=a, b=b)
            solver = sinkhorn.Sinkhorn(**sinkhorn_kwargs)
            out = solver(prob)

            converged = bool(out.converged)
            w2 = float(jnp.sqrt(jnp.clip(out.reg_ot_cost, 0.0)))

            if not converged:
                warnings.warn(f'Sinkhorn did not converge for {label}')

            print(f'{label:>20s} | {w2:10.4f} | '
                  f'{"yes" if converged else "NO":>10s} | {n_samp:8d}')
            w2_results[label] = w2
        except Exception as e:
            print(f'{label:>20s} | {"FAILED":>10s} | {"":>10s} | '
                  f'{n_samp:8d}  ({e})')

    return w2_results
