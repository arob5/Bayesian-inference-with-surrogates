# uncprop/models/vsem/experiment.py

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
    n_months = 12

    inverse_problem_settings = {
        'par_names': ['av', 'veg_init'],
        'n_windows': n_months,
        'n_days_per_window': 30,
        'observed_variable': 'lai'
    }

    surrogate_settings = {
        'design_method': 'lhc'
    }

    def __init__(self, 
                 key: PRNGKey, 
                 n_design: int, 
                 noise_sd: int,
                 n_grid: int = 50,
                 **kwargs):
        key, key_inv_prob, key_surrogate = jr.split(key, 3)
        self.inverse_problem_settings['noise_cov_tril'] = noise_sd * jnp.identity(self.n_months)
        self.surrogate_settings['n_design'] = n_design
        
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

        return self
    

    class VSEMExperiment(Experiment):
        pass
