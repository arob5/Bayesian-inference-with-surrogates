# experiments/vsem/rkpcn_analysis/reconstruct.py
"""
Reconstruct a VSEM replicate from saved experiment output.

Rebuilds the full replicate object (posterior, surrogate, grid) from the
base PRNG key and experiment metadata, so that the GP surrogate is available
for running new RKPCN variants.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    """Reconstruct a VSEMReplicate from experiment output.

    Reads the base key, derives the per-rep key using the same splitting
    logic as Experiment, and re-initializes the replicate (refitting the
    surrogate from the same PRNG state).

    Args:
        base_dir: Experiment base directory (e.g., out/vsem/).
        setup_name: Subdirectory name (e.g., 'clip_gp_N4').
        rep_idx: Replicate index.
        num_reps: Total number of replicates (for key splitting).
        surrogate_tag: 'gp' or 'clip_gp'. Inferred from setup_name if None.
        n_design: Design size. Inferred from setup_name if None.
        jitter: Surrogate jitter. Inferred from standard settings if None.

    Returns:
        VSEMReplicate with posterior, posterior_surrogate, grid, etc.
    """
    base_dir = Path(base_dir)
    rep_dir = base_dir / setup_name / f'rep{rep_idx}'

    if not rep_dir.exists():
        raise FileNotFoundError(f'Rep directory not found: {rep_dir}')

    # Infer settings from setup_name if not provided
    if surrogate_tag is None:
        # e.g., 'clip_gp_N4' -> 'clip_gp'
        parts = setup_name.rsplit('_N', 1)
        surrogate_tag = parts[0]
    if n_design is None:
        n_design = int(setup_name.rsplit('_N', 1)[1])
    if jitter is None:
        # Standard jitter settings from runner.py
        jitter_map = {4: 1e-4, 8: 1e-3, 16: 1e-2}
        jitter = jitter_map.get(n_design, 1e-4)

    # Reconstruct key
    base_key = jnp.load(base_dir / 'base_key.npy')
    base_key = jr.wrap_key_data(base_key)
    rep_keys = jr.split(base_key, num_reps)
    rep_key = rep_keys[rep_idx]
    key_setup, key_run = jr.split(rep_key)

    print(f'Reconstructing replicate: {setup_name}/rep{rep_idx}')
    print(f'  surrogate_tag={surrogate_tag}, n_design={n_design}, jitter={jitter}')

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


def load_saved_data(base_dir: Path, setup_name: str, rep_idx: int):
    """Load all saved per-rep data files.

    Returns:
        dict with keys 'samples', 'diagnostics', 'grid_densities',
        'setup_info', 'coverage' (each a dict of arrays).
        Missing files are returned as None.
    """
    rep_dir = Path(base_dir) / setup_name / f'rep{rep_idx}'
    data = {}

    for name in ['samples', 'diagnostics', 'grid_densities', 'setup_info',
                  'coverage', 'grid_info']:
        path = rep_dir / f'{name}.npz'
        if path.exists():
            data[name] = dict(jnp.load(path))
        else:
            data[name] = None

    return data
