from jax import config
config.update('jax_enable_x64', True)
from pathlib import Path
from typing import Any

import jax.random as jr
import jax.numpy as jnp

from uncprop.models.elliptic_pde.experiment import PDEReplicate
from uncprop.utils.experiment import Experiment

base_dir = Path('/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper')

# -----------------------------------------------------------------------------
# Experiment settings 
# -----------------------------------------------------------------------------

key = jr.key(85455)

# top level experiment settings
experiment_name = 'pde_experiment'
experiment_settings = {
    'name': experiment_name,
    'base_out_dir': base_dir / 'out' / experiment_name,
    'num_reps': 100,
    'base_key': key,
    'Replicate': PDEReplicate,
    'write_to_file': True,
}

# Different experimental setups
design_sizes = [4, 10, 20]

# replicate setup settings
noise_sd = 1e-3
obs_locations = jnp.array([10, 30, 60, 75])
setup_kwargs = {
    'n_design': None, # will be filled in
    'n_kl_modes': 6,
    'num_rff': 1000,
    'obs_locations': obs_locations,
    'noise_cov': noise_sd**2 * jnp.identity(len(obs_locations)),
}

def make_subdir_name(setup_kwargs, run_kwargs):
    n = setup_kwargs['n_design']
    return f'n_design_{n}'


# replicate run settings
run_kwargs = {
    'mcmc_settings': {'n_samples': 5_000, 'n_burnin': 50_000, 
                      'thin_window': 1, 'adapt_kwargs': {'gamma_exponent': 0.5}},
    'mcwmh_settings': {'n_chains': 100, 'n_samp_per_chain': 50, 
                       'n_burnin': 10_000, 'thin_window': 100,
                       'adapt_kwargs': {'gamma_exponent': 0.5}}
}



# -----------------------------------------------------------------------------
# Run experiment 
# -----------------------------------------------------------------------------

def run_pde_experiment():
    experiment = Experiment(subdir_name_fn=make_subdir_name, 
                            **experiment_settings)
    all_results = []
    all_failed_reps = []

    for n_design in design_sizes:
        setup_kwargs['n_design'] = n_design

        results, failed_reps = experiment(setup_kwargs=setup_kwargs, 
                                          run_kwargs=run_kwargs)
        all_results.append(results)
        all_failed_reps.append(failed_reps)

    return all_results, all_failed_reps


if __name__ == '__main__':
    run_pde_experiment()