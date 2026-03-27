# experiments/elliptic_pde/experiment.py
from pathlib import Path
from typing import Any, NamedTuple
import time

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.core.inverse_problem import Posterior
from uncprop.core.distribution import DistributionFromDensity
from uncprop.core.samplers import (
    _f_update_pcn_proposal,
    sample_distribution,
    init_rkpcn_kernel,
    mcmc_loop,
)
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
                 write_to_file: bool = True,
                 **kwargs):
        
        jax.clear_caches()
        if write_to_file:
            jnp.savez(out_dir / 'init_settings.npz', 
                      key_init=jr.key_data(key),
                      out_dir=out_dir,
                      n_design=n_design,
                      num_rff=num_rff,
                      design_method=design_method,
                      write_to_file=write_to_file)

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
        if write_to_file:
            jnp.savez(out_dir / 'design.npz', X=self.design.X, y=self.design.y)
            jnp.savez(out_dir / 'keys.npz', **{nm: jr.key_data(k) for nm, k in self.keys.items()})


    def __call__(self,
                 key: PRNGKey,
                 out_dir: Path,
                 rho_vals: list[float] | None = None,
                 mcmc_settings: dict[str, Any] | None = None,
                 mcwmh_settings: dict[str, Any] | None = None,
                 rkpcn_settings: dict[str, Any] | None = None,
                 **kwargs):
        
        key, key_init_mcmc, key_seed_mcmc, key_mcwmh = jr.split(key, 4)

        if rho_vals is None:
            rho_vals = [0.0, 0.9, 0.95, 0.99]
        if mcmc_settings is None:
            mcmc_settings = {'n_samples': 5000, 'n_burnin': 10_000, 'thin_window': 5}
        if mcwmh_settings is None:
            mcwmh_settings = {'n_chains': 100, 'n_samp_per_chain': 10, 'n_burnin': 10_000, 'thin_window': 100}
        if rkpcn_settings is None:
            rkpcn_settings = {'n_samples': 5000, 'n_burnin': 10_000, 'thin_window': 5}

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
        diagnostics = {}
        print('\tRunning samplers')
        for key, (dist_name, dist) in zip(mcmc_keys, dists.items()):
            jax.clear_caches()
            print(f'\t\t{dist_name}')
            mcmc_results = sample_distribution(
                key=key,
                dist=dist,
                initial_position=initial_position,
                prop_cov=0.3**2 * jnp.eye(self.posterior.dim),
                **mcmc_settings
            )

            mcmc_samp[dist_name] = mcmc_results['positions'].squeeze(1)
            mcmc_info[dist_name] = mcmc_results
            diagnostics[f'{dist_name}_accept_rate'] = mcmc_results['accept_rate']
            print(f'\t\t  accept rate: {mcmc_results["accept_rate"]:.4f}')
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)
        jnp.savez(out_dir / 'diagnostics.npz',
                  **{k: jnp.array(v) for k, v in diagnostics.items()})

        # expected posterior: MCwMH
        start_mcwmh = time.perf_counter()
        samp_mcwmh = sample_mcwmh(key=key_mcwmh,
                                  posterior_surrogate=self.posterior_surrogate,
                                  prop_cov_init=mcmc_info['mean']['prop_cov'][0],
                                  **mcwmh_settings)
        mcmc_samp['ep_mcwmh'] = samp_mcwmh.reshape(-1, self.posterior.dim)
        end_mcwmh = time.perf_counter()
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)
        diagnostics['mcwmh_time'] = end_mcwmh - start_mcwmh
        print(f'\t\tmcwmh time: {end_mcwmh - start_mcwmh:.6f} seconds')

        # rkpcn samplers
        rkpcn_output = {}
        rkpcn_prop_cov = jnp.cov(mcmc_samp['eup'], rowvar=False)

        start_rkpcn = time.perf_counter()
        for rho in rho_vals:
            jax.clear_caches()
            tag = f'rkpcn{int(rho*100)}'
            print(f'\t\t{tag} (rho={rho})')
            key, key_rkpcn = jr.split(key)

            samp_rkpcn, accept_rate = sample_rkpcn(
                key=key_rkpcn,
                posterior=self.posterior,
                surrogate_post=self.posterior_surrogate,
                initial_position=initial_position.squeeze(),
                prop_cov=rkpcn_prop_cov,
                rho=rho,
                **rkpcn_settings)
            rkpcn_output[tag] = samp_rkpcn
            diagnostics[f'{tag}_accept_rate'] = accept_rate
            print(f'\t\t  accept rate: {accept_rate:.4f}')

        end_rkpcn = time.perf_counter()
        diagnostics['rkpcn_time'] = end_rkpcn - start_rkpcn
        mcmc_samp = mcmc_samp | rkpcn_output
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)
        jnp.savez(out_dir / 'diagnostics.npz',
                  **{k: jnp.array(v) for k, v in diagnostics.items()})
        print(f'\t\trkpcn time: {end_rkpcn - start_rkpcn:.6f} seconds')

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


def sample_rkpcn(key: PRNGKey,
                 posterior: Posterior,
                 surrogate_post: PDEFwdModelGaussianSurrogate,
                 initial_position: Array,
                 prop_cov: Array,
                 rho: float,
                 n_samples: int,
                 n_burnin: int = 50_000,
                 thin_window: int = 5):
    """rk-pcn algorithm for approximate EP inference"""
    
    key_ker, key_init_state, key_samp = jr.split(key, 3)    

    # log-density as a function of target function output
    observable_to_logdensity = posterior.likelihood.observable_to_logdensity
    truncated_log_prior = DistributionFromDensity(log_dens=posterior.prior.log_density,
                                                  dim=posterior.dim, support=surrogate_post.support)
    truncated_log_prior_density = truncated_log_prior.log_density

    def log_density(f, u):
        return observable_to_logdensity(f).squeeze() + truncated_log_prior_density(u).squeeze()

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
    initial_state = init_fn(key=key_init_state,
                            initial_position=initial_position,
                            prop_cov=prop_cov)                                         
    
    # run sampler
    n_samples_total = n_burnin + thin_window * n_samples
    states, infos = mcmc_loop(key=key_samp,
                              kernel=kernel,
                              initial_state=initial_state,
                              num_samples=n_samples_total)

    samp = states.position[n_burnin:]
    accept_rate = float(jnp.mean(infos.accept_prob))

    return samp[::thin_window], accept_rate

