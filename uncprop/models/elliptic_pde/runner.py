from jax import config
config.update('jax_enable_x64', True)

import sys
import argparse
from pathlib import Path
from typing import Any
import math

import jax.random as jr
import jax.numpy as jnp

from uncprop.models.elliptic_pde.experiment import PDEReplicate
from uncprop.utils.experiment import Experiment


base_dir = Path('/projectnb/dietzelab/arober/Bayesian-inference-with-surrogates')

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
}

# Different experimental setups
design_sizes = [4, 10, 20]

# replicate setup settings
noise_sd = 1e-3
obs_locations = jnp.array([10, 30, 60, 75])
base_setup_kwargs = {
    'n_design': None, # filled per job
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
# Batching utilities for cluster array job
# -----------------------------------------------------------------------------

def get_batch(design_idx, chunk_idx, rep_chunk_size):
    """Return (n_design, rep_idx) for a given batch."""
    n_reps = experiment_settings['num_reps']

    rep_start = chunk_idx * rep_chunk_size
    rep_stop = min(n_reps, (chunk_idx + 1) * rep_chunk_size)

    if rep_start >= n_reps:
        return None, []

    return design_sizes[design_idx], list(range(rep_start, rep_stop))


def task_id_to_indices(task_id, rep_chunk_size):
    """ Map array job task ID to a chunk of experiment runs

    Let C := ceil(num_reps / rep_chunk_size). Then the total number of
    jobs is J = C * len(design_sizes). We consider the task ID in
    {0, 1, ..., J-1} and define the mapping from task ID to the chunk via:
        design_idx := task_id // C
        chunk_idx := task_id % C
    """

    n_chunks = math.ceil(experiment_settings['num_reps'] / rep_chunk_size)

    design_idx = task_id // n_chunks
    chunk_idx = task_id % n_chunks

    if design_idx >= len(design_sizes):
        raise ValueError("task_id out of range")

    return design_idx, chunk_idx


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run_task(task_id, rep_chunk_size):
    design_idx, chunk_idx = task_id_to_indices(task_id, rep_chunk_size)
    n_design, rep_idx = get_batch(design_idx, chunk_idx, rep_chunk_size)

    if not rep_idx:
        print(f"[task {task_id}] No replicates to run — exiting.")
        return

    setup_kwargs = dict(base_setup_kwargs)
    setup_kwargs['n_design'] = n_design

    print(
        f'[task {task_id}] '
        f'design_idx={design_idx}, n_design={n_design}, '
        f'chunk_idx={chunk_idx}, reps={rep_idx}'
    )

    experiment = Experiment(
        subdir_name_fn=make_subdir_name,
        **experiment_settings,
    )

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        write_to_log_file=True,
    )

    if failed_reps:
        print(f'[task {task_id}] Failed reps: {failed_reps}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--rep-chunk-size', type=int, default=10)

    args = parser.parse_args()

    run_task(args.task_id, args.rep_chunk_size)


def main_manual(n_design, rep_idx, write_to_log_file):
    setup_kwargs = dict(base_setup_kwargs)
    setup_kwargs['n_design'] = n_design

    print(
        f'[manual run] '
        f'n_design={n_design}, '
        f'reps={rep_idx}'
    )

    experiment = Experiment(
        subdir_name_fn=make_subdir_name,
        **experiment_settings,
    )

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        overwrite=True,
        write_to_log_file=write_to_log_file,
    )


if __name__ == "__main__":
    main()