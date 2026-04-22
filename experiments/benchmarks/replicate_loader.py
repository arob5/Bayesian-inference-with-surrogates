# experiments/benchmarks/replicate_loader.py
"""
Unified replicate loader for the benchmark framework.

Each experiment's ``reconstruct.py`` knows how to rebuild its own
replicate class from saved experiment output. This module dispatches
to the correct reconstruct based on the ``experiment`` field in the
benchmark config.

Replicate contract (all replicates must expose):
    .posterior            -- core.Posterior (for get_adapted_proposal)
    .posterior_surrogate  -- core.SurrogateDistribution

Optional attributes (enable experiment-specific features):
    .grid                 -- 2D comparison grid (VSEM only)
    .par_names            -- list[str] of parameter names

The :func:`get_par_names` helper returns a parameter-name list that
works uniformly across replicates (falling back to the underlying
prior's ``par_names`` if the replicate doesn't expose them).

Output-directory layout assumed by :func:`load_replicate`:
    <repo_root>/out/<experiment>/<setup>/rep<idx>/
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Dispatch
# =============================================================================

def load_replicate(experiment: str, setup: str, rep: int, base_dir=None):
    """Load a reconstructed replicate for any supported experiment.

    Parameters
    ----------
    experiment : str
        Experiment name (``'vsem'`` or ``'elliptic_pde'``).
    setup : str
        Setup subdirectory name (e.g. ``'clip_gp_N4'``, ``'n_design_10'``).
    rep : int
        Replicate index.
    base_dir : Path or None
        Experiment base directory. If None, uses
        ``<repo_root>/out/<experiment>/``.

    Returns
    -------
    (rep_obj, key_run) : (Replicate, PRNGKey)
        The reconstructed replicate plus a per-rep PRNG key suitable
        for downstream RKPCN runs (the key the Experiment driver
        originally handed to ``rep.__call__``).
    """
    if base_dir is None:
        base_dir = REPO_ROOT / 'out' / experiment
    base_dir = Path(base_dir)

    if experiment == 'vsem':
        sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'vsem'))
        from reconstruct import reconstruct_replicate
        return reconstruct_replicate(base_dir, setup, rep)

    if experiment == 'elliptic_pde':
        sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'elliptic_pde'))
        from reconstruct import reconstruct_replicate
        return reconstruct_replicate(base_dir, setup, rep)

    raise ValueError(
        f"Unknown experiment: {experiment!r}. "
        f"Supported: 'vsem', 'elliptic_pde'."
    )


# =============================================================================
# Replicate introspection helpers
# =============================================================================

def get_par_names(rep) -> list[str]:
    """Parameter names for a replicate.

    Prefers ``rep.par_names``; falls back to ``rep.posterior.prior.par_names``;
    finally returns generic ``u0, u1, ...`` labels.
    """
    if hasattr(rep, 'par_names') and rep.par_names is not None:
        return list(rep.par_names)
    prior = getattr(rep.posterior, 'prior', None)
    if prior is not None and hasattr(prior, 'par_names'):
        names = getattr(prior, 'par_names')
        if names is not None:
            return list(names)
    d = rep.posterior_surrogate.dim
    return [f'u{i}' for i in range(d)]


def rep_id(experiment: str, setup: str, rep_idx: int) -> str:
    """Canonical identifier string for a replicate, used as a subdir name.

    Examples
    --------
    >>> rep_id('vsem', 'clip_gp_N4', 0)
    'vsem_clip_gp_N4_rep0'
    """
    return f'{experiment}_{setup}_rep{rep_idx}'


def has_grid(rep) -> bool:
    """Whether the replicate exposes a 2D comparison grid (VSEM)."""
    return hasattr(rep, 'grid') and rep.grid is not None


# =============================================================================
# VSEM-specific loading for analysis (grid densities, reference samples)
# =============================================================================

def load_vsem_saved_data(base_dir, setup: str, rep: int) -> dict:
    """Load VSEM per-rep saved data files (samples, grid densities, ...).

    Returns a dict (possibly with None values for missing files). Used
    by the analyze subcommand to produce VSEM-specific plots.
    """
    sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'vsem'))
    from reconstruct import load_saved_data
    return load_saved_data(base_dir, setup, rep)
