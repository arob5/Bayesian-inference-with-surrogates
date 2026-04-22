# uncprop/core/rkpcn_multichain.py
"""
Generic RKPCN runners — single-chain and multi-chain.

These functions are surrogate-agnostic: they operate on any
``SurrogateDistribution`` that implements the standard interface
(``.support``, ``.surrogate``, ``.dim``, ``.sample_surrogate_pred``,
``.log_density_from_samples``). This module is the unified entry point
for RKPCN experiments across VSEM, the elliptic PDE, and any future
surrogate-based inverse problems.

Two runners are provided:

- :func:`run_rkpcn_chain`: single RKPCN chain with optional adaptive
  proposal tuning. Returns the full (unthinned) trace.
- :func:`run_rkpcn_multi_chain`: ``M`` RKPCN chains from diverse
  EP-aware starting positions, with R-hat-based failure / duplicate-mode
  detection and mode-level Pritchard weighting.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from uncprop.custom_types import PRNGKey, Array
from uncprop.core.surrogate import SurrogateDistribution
from uncprop.core.rkpcn import (
    RKPCNConfig,
    build_rkpcn_kernel,
    build_log_density_fn,
)
from uncprop.core.rkpcn_adaptation import (
    AdaptiveRKPCNConfig,
    build_adaptive_rkpcn_kernel,
)
from uncprop.core.samplers import mcmc_loop
from uncprop.core.chain_combiner import (
    run_multi_chain,
    compute_chain_weights,
    detect_failed_chains,
    identify_duplicate_modes,
    merge_chains_by_mode,
    combine_chains,
    select_initial_positions,
    print_multi_chain_summary,
)
from uncprop.utils.diagnostics import compute_ess


# =============================================================================
# Single-chain runner
# =============================================================================

def run_rkpcn_chain(
    key: PRNGKey,
    surrogate_post: SurrogateDistribution,
    rho: float = 0.99,
    n_u_steps: int = 1,
    prop_cov: Array | None = None,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    adaptive: bool = False,
    adapt_end: int | None = None,
    adapt_interval: int = 50,
    target_accept: float = 0.234,
    gamma_exponent: float = 0.8,
    initial_position: Array | None = None,
    label: str | None = None,
) -> dict:
    """Run a single RKPCN chain, optionally with adaptive proposal tuning.

    Parameters
    ----------
    key : PRNGKey
    surrogate_post : SurrogateDistribution
        Target surrogate posterior.
    rho : float
        pCN correlation parameter.
    n_u_steps : int
        Number of u-updates per f-update.
    prop_cov : (d, d) array or scalar or None
        Proposal covariance (or scale-of-I). ``None`` -> ``0.01 * I``.
    n_total : int
        Total macro-iterations (burn-in + post-burn-in).
    n_burnin : int
        Burn-in iterations.
    adaptive : bool
        Whether to use the adaptive kernel (scale + shape tuning).
    adapt_end, adapt_interval, target_accept, gamma_exponent :
        Adaptive-kernel settings (ignored if ``adaptive=False``).
        ``adapt_end`` defaults to ``n_burnin``.
    initial_position : (d,) array or None
        Chain start. If ``None``, drawn from the surrogate support
        via :func:`select_initial_positions` (single EP-direct draw).
    label : str or None
        Display label for logs.

    Returns
    -------
    dict with keys
        positions, logdensities, accept_probs, is_accepted,
        post_burnin, ess, accept_rate, rho, n_u_steps, label,
        n_burnin, runtime, adaptive, adapted_prop_cov_diag (if adaptive).
    """
    key_init, key_samp = jr.split(key)
    d = surrogate_post.dim
    gp = surrogate_post.surrogate
    log_density_fn = build_log_density_fn(surrogate_post)

    # Normalize prop_cov
    if prop_cov is None:
        prop_cov = 0.01 * jnp.eye(d)
    elif jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    # Initial position
    if initial_position is None:
        # Single EP-aware starting point
        initial_position = select_initial_positions(
            key_init, surrogate_post=surrogate_post,
            n_chains=1, method='ep_direct_sampling',
        )[0]
    initial_position = jnp.atleast_1d(jnp.squeeze(jnp.asarray(initial_position)))

    # Build kernel
    config = RKPCNConfig(rho=rho, n_u_steps=n_u_steps)
    if adaptive:
        adapt_cfg = AdaptiveRKPCNConfig(
            adapt_end=adapt_end if adapt_end is not None else n_burnin,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
            gamma_exponent=gamma_exponent,
        )
        init_fn, kernel_fn = build_adaptive_rkpcn_kernel(
            config, adapt_cfg, log_density_fn, gp)
    else:
        init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    state = init_fn(key_init, initial_position, prop_cov)

    if label is None:
        parts = [f'rho{int(rho*100)}']
        if n_u_steps > 1:
            parts.append(f'u{n_u_steps}')
        if adaptive:
            parts.append('adapt')
        label = '_'.join(parts)

    print(f'  Running {label} (rho={rho}, n_u_steps={n_u_steps}, '
          f'adaptive={adaptive}, n_total={n_total}, n_burnin={n_burnin})...')

    start = time.perf_counter()
    states, infos = mcmc_loop(
        key=key_samp, kernel=kernel_fn,
        initial_state=state, num_samples=n_total,
    )
    runtime = time.perf_counter() - start

    positions = np.array(states.position)
    logdensities = np.array(states.logdensity)
    accept_probs = np.array(infos.accept_prob)
    is_accepted = np.array(infos.is_accepted)

    post_burnin = positions[n_burnin:]
    ess = (compute_ess(post_burnin) if post_burnin.shape[0] > 10
           else np.zeros(d))
    accept_rate = float(np.mean(accept_probs[n_burnin:])) if n_burnin < n_total else 0.0

    print(f'    accept={accept_rate:.4f}, min_ESS={min(ess):.1f}, '
          f'n_post={post_burnin.shape[0]}, time={runtime:.1f}s')

    result = {
        'positions': positions,
        'logdensities': logdensities,
        'accept_probs': accept_probs,
        'is_accepted': is_accepted,
        'post_burnin': post_burnin,
        'ess': ess,
        'accept_rate': accept_rate,
        'rho': rho,
        'n_u_steps': n_u_steps,
        'label': label,
        'n_burnin': n_burnin,
        'runtime': runtime,
        'adaptive': adaptive,
        'multi_chain': False,
    }

    if adaptive:
        final_tril = np.array(states.proposal_tril[-1])
        adapted_cov = final_tril @ final_tril.T
        result['adapted_prop_cov_diag'] = np.diag(adapted_cov).tolist()
        print(f'    adapted prop_cov diag: {np.diag(adapted_cov)}')

    return result


# =============================================================================
# Multi-chain runner
# =============================================================================

def run_rkpcn_multi_chain(
    key: PRNGKey,
    surrogate_post: SurrogateDistribution,
    n_chains: int = 4,
    rho: float = 0.99,
    n_u_steps: int = 1,
    prop_cov: Array | None = None,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    adaptive: bool = True,
    adapt_interval: int = 50,
    target_accept: float = 0.234,
    init_method: str = 'ep_direct_sampling',
    init_n_candidates: int = 500,
    init_n_trials: int = 100,
    weight_method: str = 'pritchard',
    par_names: list[str] | None = None,
    label: str | None = None,
) -> dict:
    """Run multiple RKPCN chains from diverse EP-aware starting positions.

    Pipeline:

    1. Select ``n_chains`` initial positions via the EP-direct-sampling
       initializer (or another supported method).
    2. Run each chain sequentially with (optional) adaptive proposal.
    3. Per-chain: split-R-hat-based auto burn-in + failure detection.
    4. Cross-chain: agglomerative pairwise-R-hat mode clustering.
    5. Mode-level Pritchard weighting; pool samples with per-sample
       weights preserving mode weights.

    All diagnostics (per-chain ESS/R-hat, per-mode Pritchard weights,
    mode membership, init positions) are returned in the result dict
    for downstream analysis and plotting.

    Parameters
    ----------
    key : PRNGKey
    surrogate_post : SurrogateDistribution
    n_chains : int
    rho, n_u_steps : RKPCN kernel params (shared across chains)
    prop_cov : (d, d) initial proposal covariance. None -> 0.01*I.
    n_total, n_burnin : per-chain iteration counts
    adaptive : bool
    adapt_interval, target_accept : adaptive-kernel params
    init_method : ``'ep_direct_sampling'`` or ``'uniform_support'``
    init_n_candidates, init_n_trials : EP-init hyperparameters
    weight_method : mode weighting (``'equal'``, ``'mean_logdens'``,
        ``'pritchard'``)
    par_names : parameter names for summary printing
    label : display label

    Returns
    -------
    dict with pooled + per-chain + per-mode fields. See
    ``experiments/benchmarks/benchmark.py`` for the canonical persistence
    format.
    """
    key_init, key_chains = jr.split(key)

    d = surrogate_post.dim
    gp = surrogate_post.surrogate
    log_density_fn = build_log_density_fn(surrogate_post)

    if prop_cov is None:
        prop_cov = 0.01 * jnp.eye(d)
    elif jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    if label is None:
        label = f'rho{int(rho*100)}_{n_chains}ch'

    # ---- Initial positions ----
    print(f'  Selecting {n_chains} initial positions '
          f'(method={init_method})...')
    init_positions = select_initial_positions(
        key_init,
        surrogate_post=surrogate_post,
        n_chains=n_chains,
        method=init_method,
        n_candidates=init_n_candidates,
        n_trials=init_n_trials,
    )
    init_positions = np.array(init_positions)
    print(f'    positions: {init_positions}')

    # ---- Kernel factory ----
    config = RKPCNConfig(rho=rho, n_u_steps=n_u_steps)
    if adaptive:
        adapt_cfg = AdaptiveRKPCNConfig(
            adapt_end=n_burnin,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
        )
        def build_kernel_fn():
            return build_adaptive_rkpcn_kernel(
                config, adapt_cfg, log_density_fn, gp)
    else:
        def build_kernel_fn():
            return build_rkpcn_kernel(config, log_density_fn, gp)

    # ---- Run chains ----
    print(f'  Running {label} ({n_chains} chains, rho={rho}, '
          f'n_u_steps={n_u_steps}, adaptive={adaptive})...')
    chain_results = run_multi_chain(
        key=key_chains,
        build_kernel_fn=build_kernel_fn,
        init_positions=jnp.array(init_positions),
        n_steps=n_total,
        prop_cov=prop_cov,
        n_burnin=n_burnin,
    )

    # ---- Convergence / failure detection ----
    failed_mask, _ = detect_failed_chains(chain_results)

    # ---- Mode clustering via cross-chain R-hat ----
    mode_labels = identify_duplicate_modes(
        chain_results, failed_mask=failed_mask)

    # ---- Mode-level pooling and weighting ----
    mode_results = merge_chains_by_mode(chain_results, mode_labels)
    mode_weights = compute_chain_weights(
        mode_results, method=weight_method, n_burnin=0)

    # ---- Summary ----
    print_multi_chain_summary(
        chain_results,
        mode_results=mode_results,
        mode_weights=mode_weights,
        failed_mask=failed_mask,
        labels=mode_labels,
        par_names=par_names,
    )

    # ---- Combine ----
    if len(mode_results) > 0:
        pooled_samples, sample_weights = combine_chains(
            mode_results, mode_weights, n_burnin=0)
    else:
        pooled_samples = np.empty((0, d))
        sample_weights = np.empty(0)

    # Aggregates
    total_runtime = sum(r['runtime'] for r in chain_results)
    pooled_ess = (compute_ess(pooled_samples)
                  if pooled_samples.shape[0] > 10 else np.zeros(d))

    # Weighted-mean accept rate (per-chain effective weights)
    per_chain_weights = np.zeros(n_chains)
    for k, mr in enumerate(mode_results):
        n_in_mode = len(mr['chain_indices'])
        if n_in_mode > 0:
            for c in mr['chain_indices']:
                per_chain_weights[c] = mode_weights[k] / n_in_mode
    if per_chain_weights.sum() > 0:
        mean_accept = float(np.average(
            [r['accept_rate'] for r in chain_results],
            weights=per_chain_weights,
        ))
    else:
        mean_accept = 0.0

    # Concatenated log-densities across modes
    if mode_results:
        all_ld = np.concatenate([mr['logdensities'] for mr in mode_results])
    else:
        all_ld = np.empty(0)

    n_failed = int(failed_mask.sum())
    n_modes = len(mode_results)
    print(f'\n  Pooled: {pooled_samples.shape[0]} samples from '
          f'{n_chains - n_failed} non-failed chains '
          f'({n_modes} distinct modes), '
          f'min_ESS={min(pooled_ess):.1f}, '
          f'weighted_accept={mean_accept:.4f}, '
          f'total_time={total_runtime:.1f}s')

    return {
        # Pooled (post-burnin; burn-in already trimmed by mode-level merge)
        'post_burnin': pooled_samples,
        'sample_weights': sample_weights,
        'logdensities': all_ld,
        'positions': pooled_samples,
        'accept_probs': np.concatenate(
            [r['accept_probs'][n_burnin:] for r in chain_results]),
        'is_accepted': np.concatenate(
            [r['accept_probs'][n_burnin:] > 0.5 for r in chain_results]),
        'ess': pooled_ess,
        'accept_rate': mean_accept,
        # Variant identification
        'rho': rho,
        'n_u_steps': n_u_steps,
        'label': label,
        'n_burnin': 0,
        'runtime': total_runtime,
        # Multi-chain specifics
        'n_chains': n_chains,
        'mode_weights': mode_weights,
        'mode_results': mode_results,
        'mode_membership': [mr['chain_indices'] for mr in mode_results],
        'per_chain_results': chain_results,
        'failed_mask': failed_mask,
        'mode_labels': mode_labels,
        'init_positions': init_positions,
        'weight_method': weight_method,
        'adaptive': adaptive,
        'multi_chain': True,
    }
