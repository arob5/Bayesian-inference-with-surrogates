# uncprop/models/elliptic_pde/experiment.py
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.core.inverse_problem import Posterior
from uncprop.core.samplers import sample_distribution
from uncprop.models.elliptic_pde.inverse_problem import (
    generate_pde_inv_prob_rep,
    PDESettings,
)


class PDEReplicate(Replicate):
    """
    Replicate in PDE experiment:
        - Randomly generate synthetic data/ground truth for specified inverse problem structure
        - Randomly samples design points and fits a forward model surrogate
        - Runs MCMC for true posterior and surrogate-based approximations
    """

    def __init__(self, key: PRNGKey, **kwargs):
        key_inv_prob, key_surrogate = jr.split(key, 2)

        # default settings
        noise_sd = 1e-2
        n_kl_modes = 6
        obs_locations = jnp.array([10, 30, 60, 75])

        defaults = {
            'noise_cov' : noise_sd**2 * jnp.identity(len(obs_locations)),
            'n_kl_modes': n_kl_modes,
            'obs_locations': obs_locations,
            'settings': PDESettings()
        }

        # override defaults
        settings = {k: kwargs.get(k,v) for k, v in defaults.items()}
  
        # exact posterior
        inv_prob_info = generate_pde_inv_prob_rep(key=key_inv_prob, **settings)
        posterior = inv_prob_info[0]
        ground_truth = inv_prob_info[3]

        self.key = key
        self.posterior = posterior
        self.ground_truth = ground_truth


    def __call__(self, 
                 key: PRNGKey,
                 write_to_file: bool = True,
                 base_out_dir: Path | None = None, 
                 rep_idx: Any = None, 
                 n_mcmc: int = 5_000, 
                 **kwargs):
        
        key, key_init_mcmc, key_mcmc = jr.split(key, 3)

        if write_to_file:
            out_dir = base_out_dir / f'rep{rep_idx}'
            out_dir.mkdir()

        # exact MCMC
        positions, states, warmup_samp, prop_cov = sample_distribution(
            key=key,
            dist=self.posterior,
            initial_position=self.posterior.prior.sample(key_init_mcmc).squeeze(),
            n_samples=n_mcmc,
            n_warmup=10_000,
            thin_window=5
        )

        self.samples = {'exact': positions}

        # write results
        if write_to_file:
            jnp.savez(out_dir / 'samples.npz', **self.samples)

        return self

