from jax import config
config.update('jax_enable_x64', True)

import argparse
import math
from pathlib import Path
from typing import Any

import jax.random as jr
from experiment import VSEMReplicate
from uncprop.utils.experiment import Experiment


base_dir = Path(__file__).resolve().parents[2]  # repo root
base_out_dir = base_dir / 'out'

# -----------------------------------------------------------------------------
# Experiment settings
#   Settings here are considered fixed global state. Settings that may vary
#   across job submissions are explicitly passed via CLI arguments.
# -----------------------------------------------------------------------------

key = jr.key(9768565)

# Different experimental setups: (surrogate_tag, n_design, jitter)
gp_tags = ['gp', 'clip_gp']
design_settings = [(4, 1e-4), (8, 1e-3), (16, 1e-2)]
setups = [(tag, n, jitter) for tag in gp_tags for n, jitter in design_settings]

num_reps = 100

# Replicate setup settings (constant across setups; n_design/jitter/tag filled per job)
base_setup_kwargs: dict[str, Any] = {
    'n_grid': 50,
    'n_design': None,
    'surrogate_tag': None,
    'surrogate_settings': {'jitter': None, 'verbose': False},
}

# Replicate run settings
rkpcn_rho_vals = {'rkpcn0': 0.0, 'rkpcn90': 0.9, 'rkpcn95': 0.95, 'rkpcn99': 0.99}
run_kwargs: dict[str, Any] = {
    'rkpcn_rho_vals': rkpcn_rho_vals,
    'mcmc_settings': {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5},
    'rkpcn_settings': {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5},
}


def make_subdir_name(setup_kwargs, run_kwargs):
    return f'{setup_kwargs["surrogate_tag"]}_N{setup_kwargs["n_design"]}'


def rep_skip_fn(rep_subdir, rep_idx):
    """Skip reps that already have saved samples."""
    return (rep_subdir / 'samples.npz').exists()


# -----------------------------------------------------------------------------
# Batching utilities for cluster array job
# -----------------------------------------------------------------------------

def get_batch(setup_idx, chunk_idx, num_reps, rep_chunk_size):
    """Return (tag, n_design, jitter, rep_indices) for a given batch."""
    tag, n, jitter = setups[setup_idx]

    rep_start = chunk_idx * rep_chunk_size
    rep_stop = min(num_reps, (chunk_idx + 1) * rep_chunk_size)

    if rep_start >= num_reps:
        return None, None, None, []

    return tag, n, jitter, list(range(rep_start, rep_stop))


def task_id_to_indices(task_id, num_reps, rep_chunk_size):
    """Map array job task ID to a (setup_idx, chunk_idx) pair.

    Let C := ceil(num_reps / rep_chunk_size). Then the total number of
    jobs is J = C * len(setups). We define the mapping:
        setup_idx := task_id // C
        chunk_idx := task_id % C
    """
    n_chunks = math.ceil(num_reps / rep_chunk_size)

    setup_idx = task_id // n_chunks
    chunk_idx = task_id % n_chunks

    if setup_idx >= len(setups):
        raise ValueError(f"task_id {task_id} out of range (max {n_chunks * len(setups) - 1})")

    return setup_idx, chunk_idx


# -----------------------------------------------------------------------------
# Execution functions
# -----------------------------------------------------------------------------

def _build_experiment(experiment_name):
    return Experiment(
        name=experiment_name,
        num_reps=num_reps,
        base_out_dir=base_out_dir / experiment_name,
        base_key=key,
        Replicate=VSEMReplicate,
        subdir_name_fn=make_subdir_name,
    )


def _build_setup_kwargs(surrogate_tag, n_design, jitter):
    setup_kwargs = dict(base_setup_kwargs)
    setup_kwargs['n_design'] = n_design
    setup_kwargs['surrogate_tag'] = surrogate_tag
    setup_kwargs['surrogate_settings'] = dict(setup_kwargs['surrogate_settings'])
    setup_kwargs['surrogate_settings']['jitter'] = jitter
    return setup_kwargs


def run_task(experiment_name, task_id, num_reps, rep_chunk_size):
    """Run a batch of replicates for one setup, determined by task_id."""
    setup_idx, chunk_idx = task_id_to_indices(task_id, num_reps, rep_chunk_size)
    tag, n, jitter, rep_idx = get_batch(setup_idx, chunk_idx, num_reps, rep_chunk_size)

    if not rep_idx:
        print(f"[task {task_id}] No replicates to run — exiting.")
        return

    setup_kwargs = _build_setup_kwargs(tag, n, jitter)

    print(
        f'[task {task_id}] '
        f'experiment={experiment_name}, '
        f'num_reps={num_reps}, '
        f'setup_idx={setup_idx}, tag={tag}, n_design={n}, jitter={jitter}, '
        f'chunk_idx={chunk_idx}, reps={rep_idx}'
    )

    experiment = _build_experiment(experiment_name)

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        write_to_log_file=True,
        rep_skip_fn=rep_skip_fn,
    )


def main():
    """Entry point for cluster array job."""
    parser = argparse.ArgumentParser(description='VSEM experiment runner')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--rep-chunk-size', type=int, default=10)

    args = parser.parse_args()

    run_task(
        experiment_name=args.experiment_name,
        task_id=args.task_id,
        num_reps=num_reps,
        rep_chunk_size=args.rep_chunk_size,
    )


def main_manual(experiment_name, surrogate_tag, n_design, jitter, rep_idx,
                write_to_log_file=True, overwrite=False):
    """Manually specify setup and replicate indices to run.

    Useful for targeted reruns of failed replicates or local testing.

    Args:
        experiment_name: name of the experiment (determines output directory)
        surrogate_tag: 'gp' or 'clip_gp'
        n_design: number of design points (4, 8, or 16)
        jitter: GP jitter parameter
        rep_idx: list of replicate indices to run
        write_to_log_file: whether to write stdout to log file
        overwrite: if True, re-run reps even if output exists
    """
    setup_kwargs = _build_setup_kwargs(surrogate_tag, n_design, jitter)

    print(
        f'[manual run] '
        f'experiment={experiment_name}, '
        f'num_reps={num_reps}, '
        f'tag={surrogate_tag}, n_design={n_design}, jitter={jitter}, '
        f'reps={rep_idx}'
    )

    experiment = _build_experiment(experiment_name)

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        write_to_log_file=write_to_log_file,
        overwrite=overwrite,
        rep_skip_fn=rep_skip_fn,
    )


if __name__ == "__main__":
    main()
