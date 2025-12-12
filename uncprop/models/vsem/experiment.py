# uncprop/models/vsem/experiment.py
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.utils.grid import Grid, DensityComparisonGrid
from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep
from uncprop.models.vsem.surrogate import fit_vsem_surrogate


class VSEMReplicate(Replicate):
    """
    Replicate in VSEM experiment:
        - Randomly synthetic data/ground truth for specified inverse problem structure
        - Randomly samples design points and fits a log-posterior surrogate
        - Computes true posterior and surrogate-based approximations on 2d grid
    """

    # fixed settings across all experiments
    n_months: int = 12

    inverse_problem_settings: dict[str, Any] = {
        'par_names': ['av', 'veg_init'],
        'n_windows': n_months,
        'n_days_per_window': 30,
        'observed_variable': 'lai'
    }

    surrogate_settings: dict[str, Any] = {
        'design_method': 'lhc'
    }

    def __init__(self, 
                 key: PRNGKey, 
                 n_design: int, 
                 noise_sd: float,
                 n_grid: int = 50,
                 verbose: bool = True,
                 jitter: float = 0.0,
                 **kwargs):
        key, key_inv_prob, key_surrogate = jr.split(key, 3)
        self.inverse_problem_settings['noise_cov_tril'] = noise_sd * jnp.identity(self.n_months)
        self.surrogate_settings['n_design'] = n_design
        self.surrogate_settings['verbose'] = verbose
        self.surrogate_settings['jitter'] = jitter
        
        # exact posterior
        posterior = generate_vsem_inv_prob_rep(key=key_inv_prob,
                                               **self.inverse_problem_settings)
        
        # surrogate posterior
        surrogate_post_gp, surrogate_post_clip, fit_info = fit_vsem_surrogate(key=key_surrogate, 
                                                                              posterior=posterior,
                                                                              **self.surrogate_settings)
        
        # grid points for grid-based metrics
        grid = Grid(low=posterior.support[0],
                    high=posterior.support[1],
                    n_points_per_dim=[n_grid, n_grid],
                    dim_names=posterior.prior.par_names)
        
        self.key = key
        self.posterior = posterior
        self.surrogate_posterior_gp = surrogate_post_gp
        self.surrogate_posterior_clip_gp = surrogate_post_clip
        self.fit_info = fit_info
        self.grid = grid
        self.surrogate_pred = surrogate_post_gp.surrogate(grid.flat_grid)

    def __call__(self, surrogate_tag: str, **kwargs):
        # Note that each time this is called for a particular instance will generate the same key_ep.
        key, key_ep = jr.split(self.key, 2)
        post = self.posterior
        if surrogate_tag == 'gp':
            surr = self.surrogate_posterior_gp
        elif surrogate_tag == 'clip_gp':
            surr = self.surrogate_posterior_clip_gp
        else:
            raise ValueError(f'surrogate_tag must be `gp` or `clip_gp`; got {surrogate_tag}')

        dists = {
            'exact': post,
            'mean': surr.expected_surrogate_approx(),
            'eup': surr.expected_density_approx(),
            'ep': surr.expected_normalized_density_approx(key_ep, grid=self.grid)
        }

        density_comparison = DensityComparisonGrid(grid=self.grid, distributions=dists)
        self.density_comparison = density_comparison
        self.coverage_results = self.density_comparison.calc_coverage(baseline='exact')

        return self
    

class VSEMExperiment(Experiment):

    def collect_results(self, results, failed_reps, *args, **kwargs):
        """
        Note that in the below comments `n_reps` is the number of non-failed replicates.
        """

        # Only format non-failed iterations
        results = [rep for i, rep in enumerate(results) if i not in failed_reps]
        if len(results) == 0:
            return None

        # surrogate mean/variance predictions at grid points; each (n_reps, n_grid)
        pred_mean = jnp.vstack([rep.surrogate_pred.mean.ravel() for rep in results])
        pred_var = jnp.vstack([rep.surrogate_pred.variance.ravel() for rep in results])

        # unnormalized log-densities of each distribution over grid points; dict of (n_reps, n_grid)
        names = list(results[0].density_comparison.distributions.keys())
        log_dens_approx = {}
        for nm in names:
            log_dens_approx[nm] = jnp.vstack([
                rep.density_comparison.log_dens_grid[nm] for rep in results
            ])

        # coverage results
        log_coverage = jnp.stack(
            [rep.coverage_results[0] for rep in results], 
            axis=0
        )

        probs = results[0].coverage_results[1]
        dist_names = results[0].coverage_results[3]

        return {'pred_mean': pred_mean,
                'pred_var': pred_var,
                'log_dens_approx': log_dens_approx,
                'log_coverage': log_coverage,
                'probs': probs,
                'dist_names': dist_names}


    def save_results(self, subdir: Path, results: list, failed_reps: list, *args, **kwargs):
        results_dict = self.collect_results(results, failed_reps)

        if results_dict is None:
            print('Results list is length 0. Not saving')
            return None

        jnp.savez(subdir / 'results.npz', **results_dict)
        jnp.savez(subdir / 'logging_info.npz', failed_reps=failed_reps)
