# uncprop/core/chain_combiner.py
"""
Multi-chain MCMC: running, weighting, and combining non-mixing chains.

Provides utilities for running multiple MCMC chains (sequentially),
computing importance-like weights to account for different modes, and
combining the chains into a single weighted sample set.

The weighting follows the Pritchard (2000) heuristic: treat each chain
as exploring a different "model" and weight by an estimate of the
marginal likelihood under that model. See ``compute_chain_weights``
for details.

All functions operate on generic chain results (positions, log-densities)
and are not specific to RKPCN — they can be used with any MCMC kernel.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from uncprop.custom_types import PRNGKey, Array
from uncprop.core.samplers import mcmc_loop
from uncprop.utils.diagnostics import compute_ess


# =============================================================================
# Running multiple chains
# =============================================================================

def run_multi_chain(
    key: PRNGKey,
    build_kernel_fn: Callable,
    init_positions: Array,
    n_steps: int,
    prop_cov: Array,
    n_burnin: int = 0,
) -> list[dict]:
    """Run M independent MCMC chains sequentially.

    Each chain is initialized at a different position and run for
    ``n_steps`` iterations using a fresh kernel (rebuilt per chain
    to avoid stale JIT caches). Caches are cleared between chains
    to manage memory.

    Parameters
    ----------
    key : PRNGKey
    build_kernel_fn : callable
        ``() -> (init_fn, kernel_fn)`` — factory that builds a fresh
        kernel for each chain. Must return blackjax-compatible
        ``(init_fn, kernel_fn)`` pair.
    init_positions : Array, shape (M, d)
        Starting positions for each chain.
    n_steps : int
        Total iterations per chain (burnin + post-burnin).
    prop_cov : Array, shape (d, d)
        Proposal covariance (shared across chains, or per-chain
        if shape is (M, d, d)).
    n_burnin : int
        Burn-in iterations (for labeling; all iterations are returned).

    Returns
    -------
    list of dicts, one per chain. Each dict has:
        positions: (n_steps, d)
        logdensities: (n_steps,)
        accept_probs: (n_steps,)
        post_burnin: (n_post, d) where n_post = n_steps - n_burnin
        ess: (d,) — ESS of post-burnin samples
        accept_rate: float — mean accept rate post-burnin
        init_position: (d,)
        runtime: float
        chain_idx: int
    """
    n_chains = init_positions.shape[0]
    results = []

    for m in range(n_chains):
        jax.clear_caches()
        key, key_init, key_run = jr.split(key, 3)

        init_fn, kernel_fn = build_kernel_fn()

        # Per-chain or shared proposal
        if prop_cov.ndim == 3:
            pc = prop_cov[m]
        else:
            pc = prop_cov

        state = init_fn(key_init, init_positions[m], pc)

        start = time.perf_counter()
        states, infos = mcmc_loop(key_run, kernel_fn, state, num_samples=n_steps)
        runtime = time.perf_counter() - start

        positions = np.array(states.position)
        logdensities = np.array(states.logdensity)
        accept_probs = np.array(infos.accept_prob)

        post_burnin = positions[n_burnin:]
        ess = compute_ess(post_burnin) if post_burnin.shape[0] > 10 else np.zeros(positions.shape[1])
        accept_rate = float(np.mean(accept_probs[n_burnin:])) if n_burnin < n_steps else 0.0

        results.append({
            'positions': positions,
            'logdensities': logdensities,
            'accept_probs': accept_probs,
            'post_burnin': post_burnin,
            'ess': ess,
            'accept_rate': accept_rate,
            'init_position': np.array(init_positions[m]),
            'runtime': runtime,
            'chain_idx': m,
        })

        print(f'    chain {m+1}/{n_chains}: '
              f'accept={accept_rate:.4f}, '
              f'min_ESS={min(ess):.1f}, '
              f'time={runtime:.1f}s')

    return results


# =============================================================================
# Chain weighting
# =============================================================================

def compute_chain_weights(
    chain_results: list[dict],
    method: str = 'pritchard',
    n_burnin: int = 0,
    failed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute importance-like weights for combining non-mixing chains.

    Parameters
    ----------
    chain_results : list of dicts
        Each must have 'logdensities' array and 'post_burnin' or
        'positions' + n_burnin for extracting post-burnin log-densities.
    method : str
        - ``'equal'``: uniform weights 1/M
        - ``'mean_logdens'``: w_m proportional to exp(mean log-density)
        - ``'pritchard'``: w_m proportional to exp(mean_ld + 0.5*var_ld)
          (Pritchard 2000 heuristic, accounts for mode "width")
    n_burnin : int
        Use log-densities after this index for weight computation.
    failed_mask : array of bool, optional
        If provided, chains marked True get weight 0.

    Returns
    -------
    weights : (M,) array, non-negative, sums to 1.
    """
    M = len(chain_results)

    if method == 'equal':
        weights = np.ones(M) / M
    else:
        log_weights = np.zeros(M)

        for m, res in enumerate(chain_results):
            ld = res['logdensities'][n_burnin:]
            mean_ld = float(np.mean(ld))

            if method == 'mean_logdens':
                log_weights[m] = mean_ld
            elif method == 'pritchard':
                var_ld = float(np.var(ld))
                log_weights[m] = mean_ld + 0.5 * var_ld
            else:
                raise ValueError(f'Unknown weighting method: {method}')

        # Normalize in log space for stability
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

    # Zero out failed chains
    if failed_mask is not None:
        weights[failed_mask] = 0.0
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            # All chains failed — fall back to equal weights
            weights = np.ones(M) / M

    return weights


# =============================================================================
# Chain diagnostics
# =============================================================================

def detect_failed_chains(
    chain_results: list[dict],
    min_ess: float = 10.0,
    min_accept: float = 0.01,
) -> tuple[np.ndarray, dict]:
    """Flag chains that failed to converge.

    A chain is considered failed if its minimum ESS (across dimensions)
    is below ``min_ess`` OR its acceptance rate is below ``min_accept``.

    Parameters
    ----------
    chain_results : list of dicts (from run_multi_chain)
    min_ess : float
    min_accept : float

    Returns
    -------
    failed_mask : (M,) boolean array, True = failed
    diagnostics : dict with per-chain stats
    """
    M = len(chain_results)
    failed = np.zeros(M, dtype=bool)
    per_chain = []

    for m, res in enumerate(chain_results):
        ess = res['ess']
        acc = res['accept_rate']
        min_e = float(min(ess)) if len(ess) > 0 else 0.0

        is_failed = (min_e < min_ess) or (acc < min_accept)
        failed[m] = is_failed

        per_chain.append({
            'chain_idx': m,
            'min_ess': min_e,
            'accept_rate': acc,
            'failed': bool(is_failed),
            'reason': (
                'low ESS' if min_e < min_ess else
                'low accept' if acc < min_accept else
                'ok'
            ),
        })

    n_failed = int(failed.sum())
    print(f'  Failed chains: {n_failed}/{M}')
    for info in per_chain:
        if info['failed']:
            print(f'    chain {info["chain_idx"]}: {info["reason"]} '
                  f'(ESS={info["min_ess"]:.1f}, accept={info["accept_rate"]:.4f})')

    return failed, {'per_chain': per_chain, 'n_failed': n_failed}


def identify_duplicate_modes(
    chain_results: list[dict],
    threshold: float = 0.1,
) -> np.ndarray:
    """Cluster chains by their mean post-burnin position.

    Uses a simple greedy clustering: chains whose mean positions are
    within ``threshold`` (Euclidean distance, in normalized coordinates)
    are assigned to the same cluster.

    Parameters
    ----------
    chain_results : list of dicts (must have 'post_burnin')
    threshold : float
        Distance threshold for merging into the same cluster.

    Returns
    -------
    labels : (M,) int array, cluster label per chain.
    """
    M = len(chain_results)
    means = np.array([res['post_burnin'].mean(axis=0) for res in chain_results])

    # Use absolute coordinates for distance (don't normalize when
    # chains are close together, as normalization would amplify noise)
    means_norm = means

    labels = -np.ones(M, dtype=int)
    next_label = 0

    for m in range(M):
        if labels[m] >= 0:
            continue
        labels[m] = next_label
        for j in range(m + 1, M):
            if labels[j] >= 0:
                continue
            dist = np.linalg.norm(means_norm[m] - means_norm[j])
            if dist < threshold:
                labels[j] = next_label
        next_label += 1

    n_modes = len(set(labels))
    print(f'  Identified {n_modes} distinct modes from {M} chains')
    for lbl in range(n_modes):
        members = [m for m in range(M) if labels[m] == lbl]
        print(f'    mode {lbl}: chains {members}')

    return labels


# =============================================================================
# Combining chains
# =============================================================================

def combine_chains(
    chain_results: list[dict],
    weights: np.ndarray,
    n_burnin: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool post-burnin samples from multiple chains with weights.

    Returns all samples concatenated, with per-sample weights
    proportional to the chain weight divided by the number of
    samples in that chain.

    Parameters
    ----------
    chain_results : list of dicts (must have 'post_burnin')
    weights : (M,) array of chain weights (should sum to 1)
    n_burnin : int
        Ignored if 'post_burnin' already exists in results.

    Returns
    -------
    samples : (N_total, d) array of all post-burnin samples
    sample_weights : (N_total,) array, sums to 1
    """
    all_samples = []
    all_weights = []

    for m, res in enumerate(chain_results):
        samp = res['post_burnin']
        n_m = samp.shape[0]
        if n_m == 0 or weights[m] == 0:
            continue
        all_samples.append(samp)
        all_weights.append(np.full(n_m, weights[m] / n_m))

    samples = np.concatenate(all_samples, axis=0)
    sample_weights = np.concatenate(all_weights)
    sample_weights /= sample_weights.sum()  # renormalize

    return samples, sample_weights


# =============================================================================
# Initial position selection
# =============================================================================

def select_initial_positions(
    key: PRNGKey,
    gp,
    prior,
    n_chains: int,
    method: str = 'high_density_spread',
    n_candidates: int = 500,
    top_k: int = 50,
) -> Array:
    """Select diverse starting positions for multi-chain MCMC.

    Parameters
    ----------
    key : PRNGKey
    gp : GPJaxSurrogate
        The GP surrogate (for scoring candidates by predictive mean).
    prior : Distribution
        The prior distribution (for sampling candidates and enforcing support).
    n_chains : int
        Number of starting positions to select.
    method : str
        - ``'prior'``: sample directly from the prior
        - ``'high_density_spread'``: sample candidates, score by GP
          predictive mean, then greedily select to maximize spread
    n_candidates : int
        Number of candidate points to generate (for 'high_density_spread').
    top_k : int
        Number of top-scoring candidates to consider (for spread selection).

    Returns
    -------
    positions : (n_chains, d) array of initial positions.
    """
    key_cand, key_select = jr.split(key)

    if method == 'prior':
        return prior.sample(key_cand, n_chains)

    elif method == 'high_density_spread':
        # Step 1: Generate candidates from the prior
        candidates = prior.sample(key_cand, n_candidates)

        # Step 2: Score by GP predictive mean (= plug-in log-density)
        pred = gp(candidates)
        scores = np.array(pred.mean.ravel())

        # Step 3: Keep top-K by score
        if top_k > n_candidates:
            top_k = n_candidates
        top_idx = np.argsort(scores)[-top_k:]
        top_candidates = np.array(candidates[top_idx])

        # Step 4: Greedy farthest-point sampling for diversity
        selected = _farthest_point_sampling(top_candidates, n_chains)
        return jnp.array(selected)

    else:
        raise ValueError(f'Unknown init method: {method}')


def _farthest_point_sampling(candidates: np.ndarray, n_select: int) -> np.ndarray:
    """Greedily select n_select points maximizing minimum pairwise distance.

    Parameters
    ----------
    candidates : (K, d) array
    n_select : int

    Returns
    -------
    selected : (n_select, d) array
    """
    K, d = candidates.shape
    n_select = min(n_select, K)

    # Normalize to [0,1] for distance computation
    lo = candidates.min(axis=0)
    hi = candidates.max(axis=0)
    scale = np.where(hi - lo > 0, hi - lo, 1.0)
    cands_norm = (candidates - lo) / scale

    # Start with the point that has highest score (last in sorted order)
    selected_idx = [K - 1]
    min_dists = np.full(K, np.inf)

    for _ in range(n_select - 1):
        # Update minimum distances to selected set
        last = cands_norm[selected_idx[-1]]
        dists_to_last = np.linalg.norm(cands_norm - last, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)

        # Zero out already-selected points
        for idx in selected_idx:
            min_dists[idx] = -1.0

        # Select point farthest from all selected
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)

    return candidates[selected_idx]


# =============================================================================
# Summary printing
# =============================================================================

def print_multi_chain_summary(
    chain_results: list[dict],
    weights: np.ndarray,
    failed_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    par_names: list[str] | None = None,
):
    """Print a formatted summary of multi-chain results.

    Parameters
    ----------
    chain_results : list of per-chain result dicts
    weights : (M,) chain weights
    failed_mask : (M,) boolean, True = failed
    labels : (M,) int, cluster labels
    par_names : parameter names for display
    """
    M = len(chain_results)
    d = chain_results[0]['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{j}' for j in range(d)]

    header = (f'{"chain":>6s} | {"weight":>7s} | {"accept":>7s} | '
              f'{"min ESS":>8s} | {"failed":>7s} | {"mode":>5s} | '
              + ' '.join(f'mean({p})' for p in par_names))
    print(header)
    print('-' * len(header))

    for m, res in enumerate(chain_results):
        w = weights[m]
        acc = res['accept_rate']
        ess = res['ess']
        min_e = float(min(ess)) if len(ess) > 0 else 0.0
        fail = 'FAIL' if (failed_mask is not None and failed_mask[m]) else ''
        mode = str(labels[m]) if labels is not None else ''
        means = res['post_burnin'].mean(axis=0)
        means_str = ' '.join(f'{means[j]:10.4f}' for j in range(d))

        print(f'{m:6d} | {w:7.4f} | {acc:7.4f} | {min_e:8.1f} | '
              f'{fail:>7s} | {mode:>5s} | {means_str}')
