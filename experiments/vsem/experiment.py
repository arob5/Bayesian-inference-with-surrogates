# experiments/vsem/experiment.py
from pathlib import Path
from typing import Any, NamedTuple
from types import MappingProxyType
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.utils.grid import Grid, DensityComparisonGrid
from uncprop.core.inverse_problem import Posterior
from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep

from uncprop.models.vsem.surrogate import (
    fit_vsem_surrogate,
    VSEMPosteriorSurrogate,
    LogDensClippedGPSurrogate,
)
from uncprop.core.samplers import (
    sample_distribution,
    mcmc_loop,
    init_rkpcn_kernel,
    _f_update_pcn_proposal,
)

class VSEMReplicate(Replicate):
    """
    Replicate in VSEM experiment:
        - Randomly generate synthetic data/ground truth for specified inverse problem structure
        - Randomly samples design points and fits a log-posterior surrogate
        - Computes true posterior and surrogate-based approximations on 2d grid
    """

    _DEFAULT_INVERSE_PROBLEM_SETTINGS = MappingProxyType({
        'par_names': ('av', 'veg_init'),
        'n_windows': 12,
        'noise_cov_tril': jnp.identity(12),
        'n_days_per_window': 30,
        'observed_variable': 'lai',
    })

    _DEFAULT_SURROGATE_SETTINGS = MappingProxyType({
        'design_method': 'lhc',
        'jitter' : 0.0,
        'verbose': True,
    })

    def __init__(self, 
                 key: PRNGKey,
                 out_dir: Path,
                 n_design: int,
                 surrogate_tag: str, 
                 n_grid: int = 50,
                 inverse_problem_settings: Mapping[str, Any] | None = None,
                 surrogate_settings: Mapping[str, Any] | None = None,
                 **kwargs):
        
        key_inv_prob, key_surrogate = jr.split(key, 2)

        # Per-instance copies
        self.inverse_problem_settings = {
            **self._DEFAULT_INVERSE_PROBLEM_SETTINGS,
            **(inverse_problem_settings or {}),
        }

        self.surrogate_settings = {
            **self._DEFAULT_SURROGATE_SETTINGS,
            **(surrogate_settings or {}),
            'n_design': n_design,
            'surrogate_tag': surrogate_tag,
        }
        
        # exact posterior
        posterior = generate_vsem_inv_prob_rep(key=key_inv_prob,
                                               **self.inverse_problem_settings)
        vsem_params_dict = posterior.likelihood.model.vsem_params
        vsem_params = jnp.array(list(vsem_params_dict.values()))
        vsem_param_names = list(vsem_params_dict.keys())
        
        # surrogate posterior
        surrogate_post, fit_info = fit_vsem_surrogate(key=key_surrogate, 
                                                      posterior=posterior,
                                                      **self.surrogate_settings)
        
        # grid points for grid-based metrics
        grid = Grid(low=posterior.support[0],
                    high=posterior.support[1],
                    n_points_per_dim=[n_grid, n_grid],
                    dim_names=posterior.prior.par_names)
        
        self.key = key
        self.posterior = posterior
        self.posterior_surrogate = surrogate_post
        self.vsem_params = vsem_params
        self.vsem_param_names = vsem_param_names
        self.design = self.posterior_surrogate.surrogate.design
        self.fit_info = fit_info
        self.grid = grid
        self.surrogate_pred = self.posterior_surrogate.surrogate(grid.flat_grid)


    def __call__(self,
                 key: PRNGKey,
                 out_dir: Path,
                 rkpcn_rho_vals: dict[str, float] | None = None,
                 mcmc_settings: dict[str, Any] | None = None,
                 rkpcn_settings: dict[str, Any] | None = None,
                 **kwargs):

        out_dir = Path(out_dir)
        key, key_ep, key_init_mcmc, key_seed_mcmc, key_seed_rkpcn = jr.split(key, 5)

        if rkpcn_rho_vals is None:
            rkpcn_rho_vals = {}
        if mcmc_settings is None:
            mcmc_settings = {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5}
        if rkpcn_settings is None:
            rkpcn_settings = {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5}

        # exact and approximate posteriors
        surr = self.posterior_surrogate
        dists = {
            'exact': self.posterior,
            'mean': surr.expected_surrogate_approx(),
            'eup': surr.expected_density_approx(),
            'ep': surr.expected_normalized_density_approx(key_ep, grid=self.grid)
        }
        density_comparison = DensityComparisonGrid(grid=self.grid, distributions=dists)

        # --- Standard MCMC samplers (exact, mean, eup, ep) ---
        initial_position = self.posterior.prior.sample(key_init_mcmc)
        mcmc_keys = jr.split(key_seed_mcmc, len(dists))

        mcmc_samp = {}
        mcmc_info = {}
        diagnostics = {}
        print('\tRunning samplers')
        for mcmc_key, (dist_name, dist) in zip(mcmc_keys, dists.items()):
            jax.clear_caches()
            print(f'\t\t{dist_name}')
            mcmc_results = sample_distribution(
                key=mcmc_key,
                dist=dist,
                initial_position=initial_position,
                **mcmc_settings
            )

            mcmc_samp[dist_name] = mcmc_results['positions'].squeeze(1)
            mcmc_info[dist_name] = mcmc_results
            diagnostics[f'{dist_name}_accept_rate'] = mcmc_results['accept_rate']
            print(f'\t\t  accept rate: {mcmc_results["accept_rate"]:.4f}')

        # save standard MCMC samples (incremental save in case RKPCN fails)
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)

        # --- RKPCN samplers ---
        # Use the adapted proposal covariance from the exact posterior MCMC as
        # the u-proposal for RKPCN. In this synthetic experiment we have access
        # to the exact posterior, so using it for convenience to demonstrate
        # the behavior of rkpcn under reasonable tuning.
        # In practice, an adaptive RKPCN warmup phase could be used instead.
        # Note: in general, the EUP covariance could be used to initialize the
        #   rkpcn proposal. However, in this log-density emulation case, the 
        #   EUP covariance is typically very concentrated and not useful.
        prop_cov = mcmc_info['exact']['prop_cov']

        if len(rkpcn_rho_vals) > 0:
            rkpcn_keys = jr.split(key_seed_rkpcn, len(rkpcn_rho_vals))

            for i, alg_name in enumerate(rkpcn_rho_vals.keys()):
                jax.clear_caches()
                rho = rkpcn_rho_vals[alg_name]
                print(f'\t\t{alg_name} (rho={rho})')
                samp, accept_rate = _run_mcmc_rkpcn(
                    key=rkpcn_keys[i],
                    rho=rho,
                    posterior=self.posterior,
                    surrogate_post=surr,
                    initial_position=initial_position,
                    prop_cov=prop_cov,
                    **rkpcn_settings
                )
                mcmc_samp[alg_name] = samp
                diagnostics[f'{alg_name}_accept_rate'] = accept_rate
                print(f'\t\t  accept rate: {accept_rate:.4f}')

            # save all samples (overwrite with RKPCN included)
            jnp.savez(out_dir / 'samples.npz', **mcmc_samp)

        # --- Save diagnostics ---
        jnp.savez(out_dir / 'diagnostics.npz',
                  **{k: jnp.array(v) for k, v in diagnostics.items()})

        # --- Save grid densities per-rep ---
        jnp.savez(out_dir / 'grid_densities.npz',
                  **{nm: density_comparison.log_dens_grid[nm] for nm in dists})

        # --- Coverage (grid-based, comparing to exact posterior) ---
        coverage_results = density_comparison.calc_coverage(baseline='exact')
        jnp.savez(out_dir / 'coverage.npz',
                  log_coverage=coverage_results[0],
                  probs=coverage_results[1],
                  dist_names=coverage_results[3])

        # --- Save grid info (needed for post-hoc grid-based analysis) ---
        jnp.savez(out_dir / 'grid_info.npz',
                  low=self.grid.low,
                  high=self.grid.high,
                  n_points_per_dim=self.grid.n_points_per_dim,
                  dim_names=self.grid.dim_names)

        # --- Save setup info ---
        jnp.savez(out_dir / 'setup_info.npz',
                  vsem_params=self.vsem_params,
                  vsem_param_names=self.vsem_param_names,
                  design_x=self.design.X,
                  design_y=self.design.y.ravel(),
                  pred_mean=self.surrogate_pred.mean.ravel(),
                  pred_var=self.surrogate_pred.variance.ravel())

        return None
    

# -----------------------------------------------------------------------------
# Helper functions for running experiment
# -----------------------------------------------------------------------------

def _run_mcmc_rkpcn(key: PRNGKey,
                    posterior: Posterior,
                    surrogate_post: VSEMPosteriorSurrogate,
                    initial_position: Array,
                    prop_cov: Array,
                    rho: float,
                    n_samples: int,
                    n_burnin: int = 50_000,
                    thin_window: int = 5):
    
    key_ker, key_samp = jr.split(key)    

    # correctly enforce clipping operation and prior support
    low, high = surrogate_post.support
    upper_bound = lambda u: posterior.prior.log_density(u) + posterior.likelihood.log_density_upper_bound(u)

    if isinstance(surrogate_post, LogDensClippedGPSurrogate):
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            upper = upper_bound(u)
            lp = jnp.clip(f, max=upper)
            lp = jnp.where(jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()
    else:
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            lp = f
            lp = jnp.where(jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()

    # underlying GP model
    gp = surrogate_post.surrogate

    # settings for f update in sampler
    class UpdateInfo(NamedTuple):
        rho: float
    f_update_info = UpdateInfo(rho=rho)

    init_fn, kernel = init_rkpcn_kernel(key=key_ker,
                                        log_density=log_density,
                                        gp=gp,
                                        f_update_fn=_f_update_pcn_proposal,
                                        f_update_info=f_update_info)
    initial_state = init_fn(key_ker, jnp.squeeze(initial_position), prop_cov)

    # run sampler
    n_samples_total = n_burnin + thin_window * n_samples
    states, infos = mcmc_loop(key=key_samp,
                              kernel=kernel,
                              initial_state=initial_state,
                              num_samples=n_samples_total)

    samp = states.position[n_burnin:]
    accept_rate = float(jnp.mean(infos.accept_prob))

    return samp[::thin_window], accept_rate

# -----------------------------------------------------------------------------
# Post-hoc analysis: loading, W2 computation, diagnostics
# -----------------------------------------------------------------------------

def read_rep_samples(rep_dir: str | Path) -> dict:
    """Load MCMC samples from a replicate directory."""
    rep_dir = Path(rep_dir)
    return dict(jnp.load(rep_dir / 'samples.npz'))


def read_rep_diagnostics(rep_dir: str | Path) -> dict:
    """Load diagnostics from a replicate directory."""
    rep_dir = Path(rep_dir)
    return dict(jnp.load(rep_dir / 'diagnostics.npz'))


def _load_rep_grid(rep_dir):
    """Load the Grid and DensityComparisonGrid for a replicate.

    Tries to load grid coordinates from grid_info.npz. Falls back to
    reconstructing from the grid density array size and the known VSEM
    parameter bounds if grid_info.npz is absent (older runs).
    """
    rep_dir = Path(rep_dir)
    grid_dens = dict(jnp.load(rep_dir / 'grid_densities.npz'))

    gi_path = rep_dir / 'grid_info.npz'
    if gi_path.exists():
        gi = dict(jnp.load(gi_path))
        grid = Grid(low=gi['low'], high=gi['high'],
                    n_points_per_dim=gi['n_points_per_dim'],
                    dim_names=gi['dim_names'])
    else:
        # Fallback: infer from grid density size.  VSEM uses a 2D square
        # grid over [0,1] x [0,10] with equal points per dimension.
        import numpy as np
        n_pts = len(next(iter(grid_dens.values())))
        n_per_dim = int(np.sqrt(n_pts))
        grid = Grid(low=np.array([0.0, 0.0]), high=np.array([1.0, 10.0]),
                    n_points_per_dim=np.array([n_per_dim, n_per_dim]),
                    dim_names=['av', 'veg_init'])

    dcg = DensityComparisonGrid(grid=grid, log_dens_grid=grid_dens)
    return grid, dcg


def compute_w2_rep(rep_dir, epsilon=None, kde_bw_method=None,
                   sinkhorn_kwargs=None):
    """Compute grid-based W2 distances to EP for one replicate.

    All W2 distances are computed entirely on the grid, avoiding issues
    with sampling from diffuse grid-based densities:

      - exact/mean/eup/ep: grid-based log-densities loaded from disk
        (computed analytically during the experiment run).
      - rkpcn: MCMC samples are converted to grid-based log-densities
        via KDE (scipy.stats.gaussian_kde), then compared on the grid.

    The grid-based W2 uses OTT-JAX's Sinkhorn solver on the discrete
    grid geometry, comparing normalized probability vectors directly.
    This is more stable than the previous sample-based approach, which
    suffered from diffuse EP samples dominating the W2 computation.

    Args:
        rep_dir: path to replicate directory
        epsilon: Sinkhorn regularization (None = auto from grid geometry)
        kde_bw_method: bandwidth method for KDE (None = Scott's rule).
            Can be a scalar factor, 'scott', 'silverman', or a callable.
        sinkhorn_kwargs: kwargs for Sinkhorn solver

    Returns:
        (results_dict, epsilon) where results_dict maps method names to
        scalar W2 distances to EP
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {'threshold': 1e-6, 'max_iterations': 5000, 'lse_mode': True}

    rep_dir = Path(rep_dir)
    grid, dcg = _load_rep_grid(rep_dir)
    mcmc_samp = read_rep_samples(rep_dir)

    # Add RKPCN distributions to the grid via KDE
    for nm in mcmc_samp:
        if nm.startswith('rkpcn'):
            dcg.add_kde_density(nm, mcmc_samp[nm], bw_method=kde_bw_method)

    # Compute grid-based W2 for all distributions vs EP
    results = {}
    for nm in dcg.distribution_names:
        if nm == 'ep':
            continue
        w2 = dcg.calc_wasserstein2(nm, 'ep', epsilon=epsilon,
                                    **sinkhorn_kwargs)
        results[nm] = w2

    return results, epsilon


def summarize_wasserstein_reps(key, base_dir, subdir_name, rep_idcs,
                               output_dir=None,
                               kde_bw_method=None,
                               sinkhorn_kwargs=None,
                               # legacy args (ignored)
                               n_grid_samples=None, subsample=None):
    """
    Compute grid-based W2 distance to EP for all reps in a setup.

    All comparisons are performed entirely on the grid (see compute_w2_rep):
      - exact/mean/eup/ep have analytical grid densities
      - rkpcn samples are converted to grid densities via KDE

    Args:
        key: PRNG key (retained for API compatibility but unused)
        base_dir: experiment base directory (contains subdir_name/)
        subdir_name: e.g. 'gp_N4', 'clip_gp_N16'
        rep_idcs: list of replicate indices to process
        output_dir: if set, save results here
        kde_bw_method: bandwidth method for KDE of RKPCN samples
        sinkhorn_kwargs: kwargs for Sinkhorn solver

    Returns:
        (results_dict, epsilon) where results_dict maps method names to
        (n_reps,) arrays of W2 distances
    """
    if sinkhorn_kwargs is None:
        sinkhorn_kwargs = {'threshold': 1e-6, 'max_iterations': 5000, 'lse_mode': True}

    setup_dir = Path(base_dir) / subdir_name
    results = []
    eps = None

    if output_dir is not None:
        output_path = Path(output_dir) / f'w2_{subdir_name}.npz'
    else:
        output_path = None

    def _combine_results(res):
        keys = res[0].keys()
        return {k: jnp.stack([r[k] for r in res]) for k in keys}

    for i, rep_idx in enumerate(rep_idcs):
        try:
            print(f'Rep {rep_idx}')
            rep_dir = setup_dir / f'rep{rep_idx}'

            rep_results, eps = compute_w2_rep(
                rep_dir=rep_dir,
                epsilon=eps,
                kde_bw_method=kde_bw_method,
                sinkhorn_kwargs=sinkhorn_kwargs,
            )
            results.append(rep_results)

            if i > 0 and i % 20 == 0:
                if output_path is not None:
                    jnp.savez(output_path, **_combine_results(results))
                jax.clear_caches()
        except Exception as e:
            print(f'Failed rep {rep_idx}: {e}')

    if len(results) == 0:
        return {}, eps

    results = _combine_results(results)

    if output_path is not None:
        jnp.savez(output_path, **results)

    return results, eps


def summarize_diagnostics(base_dir, subdir_name, rep_idcs):
    """Load and aggregate diagnostics across replicates.

    Returns:
        dict mapping diagnostic names to (n_reps,) arrays
    """
    setup_dir = Path(base_dir) / subdir_name
    all_diag = []

    for rep_idx in rep_idcs:
        rep_dir = setup_dir / f'rep{rep_idx}'
        diag_path = rep_dir / 'diagnostics.npz'
        if diag_path.exists():
            all_diag.append(dict(jnp.load(diag_path)))
        else:
            print(f'Missing diagnostics for rep {rep_idx}')

    if len(all_diag) == 0:
        return {}

    keys = all_diag[0].keys()
    return {k: jnp.stack([d[k] for d in all_diag]) for k in keys}


def check_completion_status(base_dir, subdir_name, num_reps):
    """Check which replicates have completed (have samples.npz).

    Returns:
        (completed, missing) lists of rep indices
    """
    setup_dir = Path(base_dir) / subdir_name
    completed = []
    missing = []

    for i in range(num_reps):
        rep_dir = setup_dir / f'rep{i}'
        if (rep_dir / 'samples.npz').exists():
            completed.append(i)
        else:
            missing.append(i)

    print(f'{subdir_name}: {len(completed)}/{num_reps} completed, {len(missing)} missing')
    return completed, missing