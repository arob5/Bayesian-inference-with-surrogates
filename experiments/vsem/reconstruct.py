# experiments/vsem/reconstruct.py
"""
Reconstruct a VSEM replicate from saved experiment output.

Rebuilds the full replicate object (posterior, surrogate, grid) from
the base PRNG key and experiment metadata, so that the GP surrogate
is available for running benchmark variants.

Used by :mod:`experiments.benchmarks.replicate_loader`.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import VSEMReplicate


def reconstruct_replicate(
    base_dir: Path,
    setup_name: str,
    rep_idx: int,
    num_reps: int = 100,
    surrogate_tag: str | None = None,
    n_design: int | None = None,
    jitter: float | None = None,
):
    """Reconstruct a :class:`VSEMReplicate` from experiment output on disk.

    Reads the base key, derives the per-rep key using the same splitting
    logic as the Experiment driver, and re-initializes the replicate
    (which refits the surrogate deterministically from the same PRNG
    state).

    Parameters
    ----------
    base_dir : Path
        Experiment base directory, e.g. ``out/vsem/``.
    setup_name : str
        Subdirectory name, e.g. ``'clip_gp_N4'``.
    rep_idx : int
        Replicate index.
    num_reps : int
        Total number of replicates (for key splitting).
    surrogate_tag : str or None
        ``'gp'`` or ``'clip_gp'``. Inferred from ``setup_name`` if None.
    n_design : int or None
        Design size. Inferred from ``setup_name`` if None.
    jitter : float or None
        Surrogate jitter. Inferred from standard settings if None.

    Returns
    -------
    (rep, key_run) : (VSEMReplicate, PRNGKey)
        The reconstructed replicate plus a spare key for downstream runs.
    """
    base_dir = Path(base_dir)
    rep_dir = base_dir / setup_name / f'rep{rep_idx}'
    if not rep_dir.exists():
        raise FileNotFoundError(f'Rep directory not found: {rep_dir}')

    if surrogate_tag is None:
        surrogate_tag = setup_name.rsplit('_N', 1)[0]
    if n_design is None:
        n_design = int(setup_name.rsplit('_N', 1)[1])
    if jitter is None:
        jitter_map = {4: 1e-4, 8: 1e-3, 16: 1e-2}
        jitter = jitter_map.get(n_design, 1e-4)

    base_key = jnp.load(base_dir / 'base_key.npy')
    base_key = jr.wrap_key_data(base_key)
    rep_keys = jr.split(base_key, num_reps)
    rep_key = rep_keys[rep_idx]
    key_setup, key_run = jr.split(rep_key)

    print(f'Reconstructing VSEM replicate: {setup_name}/rep{rep_idx}')
    print(f'  surrogate_tag={surrogate_tag}, n_design={n_design}, '
          f'jitter={jitter}')

    rep = VSEMReplicate(
        key=key_setup,
        out_dir=rep_dir,
        n_design=n_design,
        surrogate_tag=surrogate_tag,
        surrogate_settings={'jitter': jitter},
        write_to_file=False,
    )
    print('  Done.')
    return rep, key_run


def load_saved_data(base_dir: Path, setup_name: str, rep_idx: int) -> dict:
    """Load all saved per-rep data files (samples, grid densities, ...).

    Returns a dict with keys ``samples``, ``diagnostics``,
    ``grid_densities``, ``setup_info``, ``coverage``, ``grid_info``;
    missing files are returned as ``None``.
    """
    rep_dir = Path(base_dir) / setup_name / f'rep{rep_idx}'
    data = {}
    for name in ['samples', 'diagnostics', 'grid_densities', 'setup_info',
                 'coverage', 'grid_info']:
        path = rep_dir / f'{name}.npz'
        data[name] = dict(jnp.load(path)) if path.exists() else None
    return data
