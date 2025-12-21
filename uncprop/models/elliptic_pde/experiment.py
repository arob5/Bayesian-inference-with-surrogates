# uncprop/models/elliptic_pde/experiment.py
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.core.inverse_problem import Posterior
from uncprop.core.surrogate import FwdModelGaussianSurrogate
from uncprop.core.samplers import sample_distribution
from uncprop.models.elliptic_pde.surrogate import fit_pde_surrogate
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

    def __init__(self, 
                 key: PRNGKey,
                 n_design: int,
                 design_method: str = 'lhc',
                 **kwargs):
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

        # fit surrogate
        print('\tFitting surrogate')
        design, surrogate, batchgp, opt_history = fit_pde_surrogate(key=key_surrogate,
                                                                    posterior=posterior,
                                                                    n_design=n_design,
                                                                    design_method=design_method)
        
        # surrogate-based posterior approximation
        posterior_surrogate = FwdModelGaussianSurrogate(gp=surrogate,
                                                        log_prior=posterior.prior.log_density,
                                                        y=posterior.likelihood.observation,
                                                        noise_cov_tril=posterior.likelihood.noise_cov_tril,
                                                        support=posterior.support)

        self.key = key
        self.posterior = posterior
        self.posterior_surrogate = posterior_surrogate
        self.batchgp = batchgp
        self.ground_truth = ground_truth
        self.design = design
        self.opt_history = opt_history


    def __call__(self, 
                 key: PRNGKey,
                 write_to_file: bool = True,
                 base_out_dir: Path | None = None, 
                 rep_idx: Any = None, 
                 n_mcmc: int = 5_000,
                 n_warmup: int = 10_000,
                 thin_window: int = 5, 
                 **kwargs):
        
        key, key_init_mcmc = jr.split(key)

        if write_to_file:
            out_dir = base_out_dir / f'rep{rep_idx}'
            out_dir.mkdir()

        # sampling distributions
        dists = {
            'exact': self.posterior,
            'mean': self.posterior_surrogate.expected_surrogate_approx(),
            'eup': self.posterior_surrogate.expected_density_approx()
        }

        initial_position = self.posterior.prior.sample(key_init_mcmc).squeeze()
        mcmc_keys = jr.split(key, len(dists))

        mcmc_samp = {}
        mcmc_info = {}
        print('\tRunning samplers')
        for key, (dist_name, dist) in zip(mcmc_keys, dists.items()):
            positions, states, warmup_samp, prop_cov = sample_distribution(
                key=key,
                dist=dist,
                initial_position=initial_position,
                n_samples=n_mcmc,
                n_warmup=n_warmup,
                thin_window=thin_window
            )

            mcmc_samp[dist_name] = positions
            mcmc_info[dist_name] = (states, warmup_samp, prop_cov)


        # write results
        if write_to_file:
            jnp.savez(out_dir / 'samples.npz', **mcmc_samp)

        self.samples = mcmc_samp
        self.mcmc_info = mcmc_info

        return self

