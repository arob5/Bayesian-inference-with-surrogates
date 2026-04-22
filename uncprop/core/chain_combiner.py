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
            if len(ld) == 0:
                log_weights[m] = -np.inf
                continue
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
# R-hat convergence diagnostic
# =============================================================================

def compute_split_rhat(chains: list[np.ndarray]) -> np.ndarray:
    """Compute split R-hat across a list of chains.

    For each chain, splits it in half and treats each half as an
    independent sub-chain. Then computes R-hat across all 2*M
    sub-chains using the standard formula::

        W = mean(within-chain variance)
        B = n * var(sub-chain means)
        V = ((n-1)/n) * W + B/n
        R-hat = sqrt(V/W)

    Parameters
    ----------
    chains : list of (N_m, d) arrays
        Each chain must have the same dimensionality d.
        Different chains can have different lengths N_m (each is
        split at its midpoint).

    Returns
    -------
    rhat : (d,) array of R-hat values per dimension.
        Returns NaN for dimensions where within-chain variance is zero.
    """
    if len(chains) == 0:
        return np.array([])

    d = chains[0].shape[1]

    # Split each chain at its midpoint → 2*M sub-chains
    sub_chains = []
    for chain in chains:
        n = chain.shape[0]
        half = n // 2
        if half < 2:
            # Too short to split meaningfully
            continue
        sub_chains.append(chain[:half])
        sub_chains.append(chain[half:2 * half])  # ensure equal lengths

    if len(sub_chains) < 2:
        return np.full(d, np.nan)

    M = len(sub_chains)
    # Use the minimum length among sub-chains (should all be equal after split)
    n = min(sc.shape[0] for sc in sub_chains)
    sub_chains = [sc[:n] for sc in sub_chains]

    # Stack: (M, n, d)
    stacked = np.stack(sub_chains, axis=0)

    # Per-dimension R-hat
    # Within-chain variance W (mean of per-chain sample variances)
    # Use sample variance (ddof=1)
    chain_vars = np.var(stacked, axis=1, ddof=1)  # (M, d)
    W = np.mean(chain_vars, axis=0)  # (d,)

    # Between-chain variance B = n * var(chain_means)
    chain_means = np.mean(stacked, axis=1)  # (M, d)
    B = n * np.var(chain_means, axis=0, ddof=1)  # (d,)

    # V = ((n-1)/n) * W + B/n
    V = ((n - 1) / n) * W + B / n

    # R-hat
    with np.errstate(divide='ignore', invalid='ignore'):
        rhat = np.sqrt(V / W)

    # NaN propagation for zero-variance dimensions is intentional
    return rhat


def _max_rhat(chains: list[np.ndarray]) -> float:
    """Compute max R-hat across dimensions for a set of chains.

    Returns NaN if any dimension has NaN R-hat (zero within-chain variance).
    """
    rhat_per_dim = compute_split_rhat(chains)
    if len(rhat_per_dim) == 0 or np.any(np.isnan(rhat_per_dim)):
        return float('nan')
    return float(np.max(rhat_per_dim))


# =============================================================================
# Within-chain convergence assessment
# =============================================================================

def assess_within_chain_convergence(
    positions: np.ndarray,
    rhat_threshold: float = 1.1,
    min_samples: int = 500,
    min_ess: float = 10.0,
    burnin_step_frac: float = 0.1,
    max_iterations: int = 20,
) -> dict:
    """Assess single-chain convergence via split R-hat with adaptive burn-in.

    Starts from the full set of positions and iteratively increases
    burn-in until the single-chain split R-hat (splitting the
    remaining samples in half and treating the halves as 2 sub-chains)
    is below ``rhat_threshold``, or until the number of remaining
    samples falls below ``min_samples``.

    After R-hat passes, verifies that minimum ESS across dimensions
    exceeds ``min_ess``. This catches stuck chains whose within-chain
    variance is technically non-zero but still essentially trivial.

    Parameters
    ----------
    positions : (N, d) array
        Full post-burnin chain positions (i.e., samples already past
        any initial burn-in configured at run time).
    rhat_threshold : float
        Chain considered converged if max R-hat < this value.
    min_samples : int
        Minimum number of samples to keep. Chain fails if convergence
        requires discarding more.
    min_ess : float
        Minimum ESS required after burn-in selection (guards against
        stuck chains that pass R-hat trivially).
    burnin_step_frac : float
        At each iteration, discard this fraction of remaining samples.
    max_iterations : int
        Maximum burn-in adjustment iterations.

    Returns
    -------
    dict with keys:
        converged : bool
        n_discarded : int — number of additional samples discarded
        n_kept : int — samples remaining after burn-in
        rhat : float — final max R-hat (NaN if stuck)
        ess_min : float — minimum ESS across dimensions after burn-in
        n_iterations : int — burn-in adjustment iterations
        fail_reason : str or None
    """
    N = positions.shape[0]
    n_discarded = 0

    for iteration in range(max_iterations + 1):
        remaining = N - n_discarded
        if remaining < min_samples:
            return {
                'converged': False,
                'n_discarded': n_discarded,
                'n_kept': remaining,
                'rhat': float('nan'),
                'ess_min': 0.0,
                'n_iterations': iteration,
                'fail_reason': 'rhat_not_converged',
            }

        current = positions[n_discarded:]
        rhat = _max_rhat([current])

        if np.isnan(rhat):
            return {
                'converged': False,
                'n_discarded': n_discarded,
                'n_kept': remaining,
                'rhat': float('nan'),
                'ess_min': 0.0,
                'n_iterations': iteration,
                'fail_reason': 'nan_rhat',
            }

        if rhat < rhat_threshold:
            # Converged — check ESS as final sanity check
            ess = compute_ess(current)
            ess_min = float(np.min(ess)) if len(ess) > 0 else 0.0

            if ess_min < min_ess:
                return {
                    'converged': False,
                    'n_discarded': n_discarded,
                    'n_kept': remaining,
                    'rhat': float(rhat),
                    'ess_min': ess_min,
                    'n_iterations': iteration,
                    'fail_reason': 'low_ess',
                }

            return {
                'converged': True,
                'n_discarded': n_discarded,
                'n_kept': remaining,
                'rhat': float(rhat),
                'ess_min': ess_min,
                'n_iterations': iteration,
                'fail_reason': None,
            }

        # Not converged — discard more
        step = max(1, int(burnin_step_frac * remaining))
        n_discarded += step

    # Exhausted iterations without converging
    return {
        'converged': False,
        'n_discarded': n_discarded,
        'n_kept': N - n_discarded,
        'rhat': float(rhat),
        'ess_min': 0.0,
        'n_iterations': max_iterations,
        'fail_reason': 'rhat_not_converged',
    }


def detect_failed_chains(
    chain_results: list[dict],
    rhat_threshold: float = 1.1,
    min_samples: int = 500,
    min_ess: float = 10.0,
    burnin_step_frac: float = 0.1,
    max_iterations: int = 20,
) -> tuple[np.ndarray, dict]:
    """Assess per-chain convergence and flag failed chains.

    For each chain, runs ``assess_within_chain_convergence`` to adaptively
    select burn-in based on split R-hat. Chains that fail to converge
    are flagged, and the ``post_burnin`` / ``logdensities`` / ``ess``
    fields of the chain result dict are updated to reflect the
    auto-selected burn-in.

    Failure reasons:
      - 'nan_rhat': within-chain variance is essentially zero (stuck)
      - 'rhat_not_converged': R-hat stays above threshold after max burn-in
      - 'low_ess': R-hat passes but ESS < min_ess (near-stuck)

    Parameters
    ----------
    chain_results : list of dicts (from run_multi_chain)
        Each dict is modified in place: 'post_burnin', 'logdensities',
        'ess' are updated to reflect the auto-selected burn-in.
    rhat_threshold : float
    min_samples : int
    min_ess : float
    burnin_step_frac : float
    max_iterations : int

    Returns
    -------
    failed_mask : (M,) boolean array, True = failed
    diagnostics : dict with per-chain stats
    """
    M = len(chain_results)
    failed = np.zeros(M, dtype=bool)
    per_chain = []

    for m, res in enumerate(chain_results):
        positions = res['post_burnin']

        assessment = assess_within_chain_convergence(
            positions,
            rhat_threshold=rhat_threshold,
            min_samples=min_samples,
            min_ess=min_ess,
            burnin_step_frac=burnin_step_frac,
            max_iterations=max_iterations,
        )

        # Update chain result to reflect auto-selected burn-in
        if assessment['n_discarded'] > 0:
            res['post_burnin'] = positions[assessment['n_discarded']:]
            if 'logdensities' in res and len(res['logdensities']) >= positions.shape[0]:
                # logdensities is typically (n_total,) — trim in sync with post_burnin
                ld = res['logdensities']
                # Offset from end (post_burnin was last N_post samples of logdensities)
                N_post = positions.shape[0]
                ld_post = ld[-N_post:]
                res['logdensities_post_burnin'] = ld_post[assessment['n_discarded']:]
            else:
                res['logdensities_post_burnin'] = None

            # Recompute ESS on kept samples
            kept = res['post_burnin']
            res['ess'] = (compute_ess(kept) if kept.shape[0] > 10
                          else np.zeros(kept.shape[1]))
        else:
            # Still set logdensities_post_burnin for consistency
            if 'logdensities' in res and len(res['logdensities']) >= positions.shape[0]:
                N_post = positions.shape[0]
                res['logdensities_post_burnin'] = res['logdensities'][-N_post:]
            else:
                res['logdensities_post_burnin'] = None

        # Store assessment in the chain result
        res['convergence'] = assessment

        failed[m] = not assessment['converged']

        per_chain.append({
            'chain_idx': m,
            'converged': assessment['converged'],
            'rhat': assessment['rhat'],
            'ess_min': assessment['ess_min'],
            'n_discarded': assessment['n_discarded'],
            'n_kept': assessment['n_kept'],
            'fail_reason': assessment['fail_reason'],
        })

    n_failed = int(failed.sum())
    print(f'  Convergence assessment: {n_failed}/{M} chains failed')
    for info in per_chain:
        status = 'FAIL' if not info['converged'] else 'ok'
        rhat_str = (f'{info["rhat"]:.3f}' if not np.isnan(info['rhat'])
                    else 'NaN')
        reason_str = f' [{info["fail_reason"]}]' if info['fail_reason'] else ''
        print(f'    chain {info["chain_idx"]}: {status} '
              f'rhat={rhat_str}, ess_min={info["ess_min"]:.1f}, '
              f'discarded={info["n_discarded"]}, kept={info["n_kept"]}'
              f'{reason_str}')

    return failed, {'per_chain': per_chain, 'n_failed': n_failed}


# =============================================================================
# Mode identification via agglomerative R-hat merging
# =============================================================================

def identify_duplicate_modes(
    chain_results: list[dict],
    rhat_threshold: float = 1.1,
    failed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cluster chains by pairwise cross-chain split R-hat.

    Uses agglomerative merging. This function assumes each individual
    chain has already passed within-chain convergence (typically via
    ``detect_failed_chains``) — i.e., each chain can be considered a
    valid local sample. The merging here asks a different question:
    are two chains sampling from the same distribution?

    Algorithm:
    1. Start with each non-failed chain as its own cluster.
    2. For all pairs of clusters, compute cross-chain split R-hat
       (splitting each chain in the cluster and combining across clusters).
    3. Find the pair with lowest R-hat. If < ``rhat_threshold``,
       merge them. Otherwise stop.
    4. Repeat until no pair can be merged.

    Failed chains (marked by ``failed_mask``) are assigned label=-1
    and excluded from merging.

    Parameters
    ----------
    chain_results : list of dicts (must have 'post_burnin')
    rhat_threshold : float
        Threshold below which chains are considered to be sampling
        from the same distribution.
    failed_mask : (M,) bool array, optional
        Chains marked True are excluded from merging (label=-1).

    Returns
    -------
    labels : (M,) int array, cluster label per chain (-1 = failed).
    """
    M = len(chain_results)
    if failed_mask is None:
        failed_mask = np.zeros(M, dtype=bool)

    # Initialize each non-failed chain as its own cluster
    labels = -np.ones(M, dtype=int)
    active_chains = [m for m in range(M) if not failed_mask[m]]

    if len(active_chains) == 0:
        print(f'  No non-failed chains to cluster')
        return labels

    # Each cluster is a list of chain indices
    clusters = {m: [m] for m in active_chains}

    while len(clusters) > 1:
        # Compute pairwise R-hat across current clusters
        cluster_keys = list(clusters.keys())
        best_pair = None
        best_rhat = float('inf')

        for i, ki in enumerate(cluster_keys):
            for kj in cluster_keys[i + 1:]:
                # Build list of chains from both clusters
                chains_i = [chain_results[c]['post_burnin']
                            for c in clusters[ki]]
                chains_j = [chain_results[c]['post_burnin']
                            for c in clusters[kj]]
                rhat = _max_rhat(chains_i + chains_j)

                if np.isnan(rhat):
                    continue
                if rhat < best_rhat:
                    best_rhat = rhat
                    best_pair = (ki, kj)

        if best_pair is None or best_rhat >= rhat_threshold:
            break

        # Merge
        ki, kj = best_pair
        clusters[ki].extend(clusters[kj])
        del clusters[kj]

    # Assign final labels
    for new_label, (key, members) in enumerate(clusters.items()):
        for c in members:
            labels[c] = new_label

    n_modes = len(clusters)
    n_failed = int(failed_mask.sum())
    n_active = M - n_failed
    print(f'  Identified {n_modes} distinct modes from {n_active} '
          f'non-failed chains ({n_failed} failed excluded)')
    for lbl, (key, members) in enumerate(clusters.items()):
        print(f'    mode {lbl}: chains {members}')

    return labels


# =============================================================================
# Mode-level merging and weighting
# =============================================================================

def merge_chains_by_mode(
    chain_results: list[dict],
    mode_labels: np.ndarray,
) -> list[dict]:
    """Concatenate per-chain samples into per-mode sample sets.

    Chains assigned to the same mode (same label) have their
    ``post_burnin`` samples and ``logdensities_post_burnin`` arrays
    concatenated. Chains with label -1 (failed) are excluded.

    The resulting per-mode dicts are suitable inputs to
    ``compute_chain_weights`` — they have the same interface as
    chain_results (with 'post_burnin' and 'logdensities' keys),
    but each represents a distinct mode rather than an individual
    chain.

    Parameters
    ----------
    chain_results : list of M dicts with keys 'post_burnin',
        'logdensities_post_burnin' (preferred) or 'logdensities'.
    mode_labels : (M,) int array with mode assignment per chain.
        Value -1 indicates a failed chain (excluded).

    Returns
    -------
    mode_results : list of K dicts (K = number of distinct non-failed modes).
        Each dict has:
            label : int (the mode label)
            chain_indices : list[int] of contributing chain indices
            post_burnin : (N_mode, d) concatenated samples
            logdensities : (N_mode,) concatenated log-densities
            n_samples : int
    """
    # Get distinct non-failed modes in label order
    distinct_modes = sorted(set(int(lbl) for lbl in mode_labels if lbl >= 0))

    mode_results = []
    for lbl in distinct_modes:
        member_idx = [m for m, l in enumerate(mode_labels) if int(l) == lbl]
        pb_pieces = []
        ld_pieces = []
        for m in member_idx:
            res = chain_results[m]
            pb_pieces.append(res['post_burnin'])
            # Prefer logdensities_post_burnin (trimmed) if available
            if res.get('logdensities_post_burnin') is not None:
                ld_pieces.append(res['logdensities_post_burnin'])
            elif 'logdensities' in res:
                # Fall back to using last N_post entries
                N_post = res['post_burnin'].shape[0]
                ld_pieces.append(res['logdensities'][-N_post:])

        pb_merged = np.concatenate(pb_pieces, axis=0) if pb_pieces else np.empty((0, 2))
        ld_merged = np.concatenate(ld_pieces) if ld_pieces else np.empty(0)

        mode_results.append({
            'label': lbl,
            'chain_indices': member_idx,
            'post_burnin': pb_merged,
            'logdensities': ld_merged,
            'n_samples': pb_merged.shape[0],
        })

    return mode_results


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

    This function operates at the **chain** level — one weight per
    chain. For mode-level weighting, first build per-mode results
    via ``merge_chains_by_mode`` and pass those here (where each
    "chain" dict is actually a merged mode).

    Parameters
    ----------
    chain_results : list of dicts (must have 'post_burnin')
    weights : (M,) array of weights (should sum to 1)
    n_burnin : int
        Ignored if 'post_burnin' already exists in results.

    Returns
    -------
    samples : (N_total, d) array of all post-burnin samples
    sample_weights : (N_total,) array, sums to 1.
        Within each input "chain" dict, samples are weighted equally:
        each gets weight[m] / n_m.
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
    mode_results: list[dict] | None = None,
    mode_weights: np.ndarray | None = None,
    failed_mask: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    par_names: list[str] | None = None,
):
    """Print a formatted summary of multi-chain results.

    Shows two sections:
      1. **Per-chain diagnostics**: individual chain statistics including
         R-hat, ESS, burn-in discarded, failure status, mode assignment.
      2. **Per-mode summary**: one row per distinct mode with the Pritchard
         weight, number of contributing chains, total samples, and mode
         mean log-density.

    Parameters
    ----------
    chain_results : list of per-chain result dicts
    mode_results : list of per-mode dicts (from merge_chains_by_mode)
    mode_weights : (K,) mode weights
    failed_mask : (M,) boolean, True = failed
    labels : (M,) int, mode label per chain (-1 = failed)
    par_names : parameter names for display
    """
    M = len(chain_results)
    d = chain_results[0]['post_burnin'].shape[1]
    if par_names is None:
        par_names = [f'u{j}' for j in range(d)]

    # ---- Per-chain section ----
    print('\n  Per-chain diagnostics:')
    header = (f'  {"chain":>6s} | {"accept":>7s} | {"min ESS":>8s} | '
              f'{"rhat":>6s} | {"disc.":>7s} | {"status":>8s} | '
              f'{"mode":>5s} | {"n":>7s} | '
              + ' '.join(f'mean({p})' for p in par_names))
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for m, res in enumerate(chain_results):
        acc = res['accept_rate']
        ess = res['ess']
        min_e = float(min(ess)) if len(ess) > 0 else 0.0

        is_failed = (failed_mask is not None and bool(failed_mask[m]))
        conv = res.get('convergence', {})
        reason = conv.get('fail_reason')
        if is_failed:
            status = reason if reason else 'FAIL'
        else:
            status = 'ok'

        mode = str(int(labels[m])) if labels is not None else ''
        n_samp = res['post_burnin'].shape[0]
        means = res['post_burnin'].mean(axis=0)
        means_str = ' '.join(f'{means[j]:10.4f}' for j in range(d))

        rhat = conv.get('rhat', float('nan'))
        rhat_str = f'{rhat:.3f}' if not np.isnan(rhat) else 'NaN'
        n_disc = conv.get('n_discarded', 0)

        print(f'  {m:6d} | {acc:7.4f} | {min_e:8.1f} | '
              f'{rhat_str:>6s} | {n_disc:7d} | {status:>8s} | '
              f'{mode:>5s} | {n_samp:7d} | {means_str}')

    # ---- Per-mode section ----
    if mode_results is not None and mode_weights is not None:
        print('\n  Per-mode summary:')
        header = (f'  {"mode":>5s} | {"weight":>7s} | {"chains":>15s} | '
                  f'{"n_samples":>10s} | {"mean_logd":>10s} | '
                  f'{"std_logd":>9s} | '
                  + ' '.join(f'mean({p})' for p in par_names))
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for k, mr in enumerate(mode_results):
            lbl = mr['label']
            chains_str = ','.join(str(c) for c in mr['chain_indices'])
            n_samp = mr['n_samples']
            w = mode_weights[k]
            ld = mr.get('logdensities')
            if ld is not None and len(ld) > 0:
                mean_ld = float(np.mean(ld))
                std_ld = float(np.std(ld))
            else:
                mean_ld = float('nan')
                std_ld = float('nan')
            means = mr['post_burnin'].mean(axis=0)
            means_str = ' '.join(f'{means[j]:10.4f}' for j in range(d))

            print(f'  {lbl:5d} | {w:7.4f} | {chains_str:>15s} | '
                  f'{n_samp:10d} | {mean_ld:10.2f} | '
                  f'{std_ld:9.3f} | {means_str}')
