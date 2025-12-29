# uncprop/models/elliptic_pde/experiment.py
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.core.inverse_problem import Posterior
from uncprop.core.samplers import sample_distribution
from uncprop.models.elliptic_pde.surrogate import fit_pde_surrogate, PDEFwdModelGaussianSurrogate
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
                 out_dir: Path,
                 n_design: int,
                 num_rff: int,
                 design_method: str = 'lhc',
                 **kwargs):
        
        jax.clear_caches()
        key_inv_prob, key_surrogate, key_rff = jr.split(key, 3)

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
        posterior_surrogate = PDEFwdModelGaussianSurrogate(gp=surrogate,
                                                           batchgp=batchgp,
                                                           posterior=posterior,
                                                           num_rff=num_rff,
                                                           key_rff=key_rff,
                                                           log_prior=posterior.prior.log_density,
                                                           y=posterior.likelihood.observation,
                                                           noise_cov_tril=posterior.likelihood.noise_cov_tril,
                                                           support=posterior.prior.truncated_support)

        self.keys = {'init': key, 'inv_prob': key_inv_prob, 'surrogate': key_surrogate, 'rff': key_rff}
        self.posterior = posterior
        self.posterior_surrogate = posterior_surrogate
        self.ground_truth = ground_truth
        self.design = design
        self.opt_history = opt_history

        # save to disk
        jnp.savez(out_dir / 'design.npz', X=self.design.X, y=self.design.y)


    def __call__(self, 
                 key: PRNGKey,
                 out_dir: Path,
                 mcmc_settings: dict[str, Any] | None = None,
                 mcwmh_settings: dict[str, Any] | None = None,  
                 **kwargs):
        
        key, key_init_mcmc, key_seed_mcmc, key_mcwmh = jr.split(key, 4)

        if mcmc_settings is None:
            mcmc_settings = {'n_samples': 5000, 'n_burnin': 10_000, 'thin_window': 5}
        if mcwmh_settings is None:
            mcwmh_settings = {'n_chains': 100, 'n_samp_per_chain': 10, 'n_burnin': 10_000, 'thin_window': 100}

        # sampling distributions
        dists = {
            'exact': self.posterior,
            'mean': self.posterior_surrogate.expected_surrogate_approx(),
            'eup': self.posterior_surrogate.expected_density_approx()
        }

        initial_position = self.posterior.prior.sample(key_init_mcmc)
        mcmc_keys = jr.split(key_seed_mcmc, len(dists))

        mcmc_samp = {}
        mcmc_info = {}
        print('\tRunning samplers')
        for key, (dist_name, dist) in zip(mcmc_keys, dists.items()):
            mcmc_results = sample_distribution(
                key=key,
                dist=dist,
                initial_position=initial_position,
                prop_cov=0.3**2 * jnp.eye(self.posterior.dim),
                **mcmc_settings
            )

            mcmc_samp[dist_name] = mcmc_results['positions'].squeeze(1)
            mcmc_info[dist_name] = mcmc_results

        # expected posterior: MCwMH
        samp_mcwmh = sample_mcwmh(key=key_mcwmh,
                                  posterior_surrogate=self.posterior_surrogate,
                                  prop_cov_init=mcmc_info['mean']['prop_cov'][0],
                                  **mcwmh_settings)
        mcmc_samp['ep_mcwmh'] = samp_mcwmh.reshape(-1, self.posterior.dim)

        # write results
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)
        jnp.savez(out_dir / 'keys.npz', **{nm: jr.key_data(k) for nm, k in self.keys.items()})

        return None


def sample_mcwmh(key: PRNGKey,
                 posterior_surrogate: PDEFwdModelGaussianSurrogate,
                 n_chains: int,
                 n_samp_per_chain: int,
                 n_burnin: int,
                 thin_window: int,
                 prop_cov_init: Array | None = None,
                 adapt_kwargs=None):
    """Monte Carlo within Metropolis-Hastings approximation of the expected posterior"""

    key_seed_trajectory, key_seed_mcmc, key_init_positions = jr.split(key, 3)

    trajectory_keys = jr.split(key_seed_trajectory, n_chains)
    mcmc_keys = jr.split(key_seed_mcmc, n_chains)

    samp = jnp.zeros((n_chains, n_samp_per_chain, posterior_surrogate.dim))
    initial_positions = posterior_surrogate.posterior.prior.sample(key_init_positions, n_chains)

    for i in range(n_chains):
        post_traj = posterior_surrogate.sample_trajectory(trajectory_keys[i])

        results = sample_distribution(
            key=mcmc_keys[i],
            dist=post_traj,
            initial_position=initial_positions[i:i+1],
            n_samples=n_samp_per_chain,
            n_burnin=n_burnin,
            thin_window=thin_window,
            prop_cov=prop_cov_init,
            adapt_kwargs=adapt_kwargs
        )

        samp = samp.at[i].set(
            results['positions'].squeeze(1)
        )

    return samp


# -----------------------------------------------------------------------------
# Helper functions for post-run analysis/plotting
# -----------------------------------------------------------------------------