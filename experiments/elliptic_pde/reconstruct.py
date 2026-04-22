# experiments/elliptic_pde/reconstruct.py
"""
Reconstruct an elliptic-PDE replicate from saved experiment output.

Rebuilds the PDEReplicate (posterior, forward-model surrogate) from
the base PRNG key and the saved init settings, so that the surrogate
is available for running benchmark variants.

Used by :mod:`experiments.benchmarks.replicate_loader`.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import PDEReplicate

# Runner-level settings that are not saved in init_settings.npz but are
# required to reproduce a replicate. These mirror
# experiments/elliptic_pde/runner.py. If runner.py changes, update here.
_DEFAULT_NOISE_SD = 1e-3
_DEFAULT_N_KL_MODES = 6
_DEFAULT_OBS_LOCATIONS = jnp.array([10, 30, 60, 75])
_DEFAULT_BASE_KEY_SEED = 85455


def reconstruct_replicate(
    base_dir: Path,
    setup_name: str,
    rep_idx: int,
    num_reps: int = 100,
    n_design: int | None = None,
    num_rff: int = 1000,
    design_method: str = 'lhc',
    noise_sd: float = _DEFAULT_NOISE_SD,
    n_kl_modes: int = _DEFAULT_N_KL_MODES,
    obs_locations=None,
    base_key_seed: int = _DEFAULT_BASE_KEY_SEED,
):
    """Reconstruct a :class:`PDEReplicate` from experiment output on disk.

    The reconstruction replays the same PRNG trajectory that the
    Experiment driver used, re-running the replicate's ``__init__``
    with ``write_to_file=False`` (no side effects).

    Because ``PDEReplicate.__init__`` does not persist all setup
    kwargs to disk (only ``n_design``, ``num_rff``, ``design_method``
    land in ``init_settings.npz``), the remaining settings
    (``noise_sd``, ``n_kl_modes``, ``obs_locations``) are assumed to
    match the defaults used in ``experiments/elliptic_pde/runner.py``.
    If a replicate was produced with non-default settings, pass them
    explicitly.

    Parameters
    ----------
    base_dir : Path
        Experiment base dir, e.g. ``out/elliptic_pde/``.
    setup_name : str
        Subdirectory name, e.g. ``'n_design_10'``.
    rep_idx : int
    num_reps : int
        Total replicates for this experiment (for key splitting).
    n_design : int or None
        Inferred from ``setup_name`` (``'n_design_<N>'``) if None.
    num_rff, design_method : PDE surrogate hyperparameters.
    noise_sd, n_kl_modes, obs_locations : inverse-problem settings.
    base_key_seed : int
        Seed for the base PRNG key (matches runner.py's ``jr.key(...)``).

    Returns
    -------
    (rep, key_run) : (PDEReplicate, PRNGKey)
    """
    base_dir = Path(base_dir)
    rep_dir = base_dir / setup_name / f'rep{rep_idx}'
    if not rep_dir.exists():
        raise FileNotFoundError(f'Rep directory not found: {rep_dir}')

    if n_design is None:
        # setup_name format: "n_design_<N>"
        n_design = int(setup_name.rsplit('_', 1)[1])

    if obs_locations is None:
        obs_locations = _DEFAULT_OBS_LOCATIONS

    noise_cov = noise_sd ** 2 * jnp.identity(len(obs_locations))

    # Rebuild base key (matches runner.py: base_key = jr.key(85455))
    base_key = jr.key(base_key_seed)
    rep_keys = jr.split(base_key, num_reps)
    rep_key = rep_keys[rep_idx]
    _, key_run = jr.split(rep_key)

    print(f'Reconstructing PDE replicate: {setup_name}/rep{rep_idx}')
    print(f'  n_design={n_design}, num_rff={num_rff}, '
          f'design_method={design_method}')
    print(f'  noise_sd={noise_sd}, n_kl_modes={n_kl_modes}, '
          f'obs_locations={obs_locations}')

    rep = PDEReplicate(
        key=rep_key,
        out_dir=rep_dir,
        n_design=n_design,
        num_rff=num_rff,
        design_method=design_method,
        noise_cov=noise_cov,
        n_kl_modes=n_kl_modes,
        obs_locations=obs_locations,
        write_to_file=False,
    )
    print('  Done.')
    return rep, key_run
