# uncprop/models/vsem/experiment.py

import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey
from uncprop.utils.experiment import Replicate
from uncprop.utils.grid import Grid, DensityComparisonGrid
from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep
from uncprop.models.vsem.surrogate import fit_vsem_surrogate


class VSEMReplicate(Replicate):

    # fixed settings across all experiments
    noise_sd = 0.1
    n_months = 24

    inverse_problem_settings = {
        'par_names': ['kext', 'av'],
        'n_windows': n_months,
        'n_days_per_window': 30,
        'observed_variable': 'lai',
        'noise_cov_tril': noise_sd * jnp.identity(n_months)
    }

    surrogate_settings = {
        'design_method': 'lhc'
    }

    def __init__(self, key: PRNGKey, n_design: int, n_grid: int = 50, **kwargs):
        key, key_inv_prob, key_surrogate = jr.split(key, 3)
        self.surrogate_settings['n_design'] = n_design
        
        # exact posterior
        posterior = generate_vsem_inv_prob_rep(key=key_inv_prob,
                                               **self.inverse_problem_settings)
        
        # surrogate posterior
        surrogate_posterior, fit_info = fit_vsem_surrogate(key=key_surrogate, 
                                                           posterior=posterior,
                                                           **self.surrogate_settings)
        
        # grid points for grid-based metrics
        grid = Grid(low=posterior.support[0],
                    high=posterior.support[1],
                    n_points_per_dim=[n_grid, n_grid],
                    dim_names=posterior.prior.par_names)
        
        self.key = key
        self.posterior = posterior
        self.surrogate_posterior = surrogate_posterior
        self.fit_info = fit_info
        self.grid = grid


    def __call__(self, **kwargs):        
        key, key_ep = jr.split(self.key, 2)
        post = self.posterior
        surr = self.surrogate_posterior

        dists = {
            'exact': post,
            'mean': surr.expected_surrogate_approx(),
            'eup': surr.expected_density_approx(),
            'ep': surr.expected_normalized_density_approx(key_ep, grid=self.grid)
        }

        density_comparison = DensityComparisonGrid(grid=self.grid, distributions=dists)
        self.density_comparison = density_comparison

        return self