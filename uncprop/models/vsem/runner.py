from jax import config
config.update('jax_enable_x64', True)
from pathlib import Path

import jax.random as jr
from uncprop.models.vsem.experiment import VSEMReplicate, VSEMExperiment

base_dir = Path('/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper')


# -----------------------------------------------------------------------------
# Experiment settings 
# -----------------------------------------------------------------------------

key = jr.key(9768565)
setup_kwargs = {'n_grid': 50, 'n_design': 8, 'noise_sd': 1.0, 'verbose': False}
num_reps = 3
experiment_name = 'vsem'
out_dir = base_dir / 'out' / experiment_name

def _make_subdir_name(setup_kwargs, run_kwargs):
    return f'{run_kwargs['surrogate_tag']}_N{setup_kwargs['n_design']}'


# -----------------------------------------------------------------------------
# Run experiment 
# -----------------------------------------------------------------------------

experiment = VSEMExperiment(name=experiment_name,
                            num_reps=num_reps,
                            base_out_dir=out_dir,
                            base_key=key,
                            Replicate=VSEMReplicate,
                            subdir_name_fn=_make_subdir_name)

experiment(run_kwargs={'surrogate_tag': 'gp'}, 
           setup_kwargs=setup_kwargs)

experiment(run_kwargs={'surrogate_tag': 'clip_gp'}, 
           setup_kwargs=setup_kwargs)