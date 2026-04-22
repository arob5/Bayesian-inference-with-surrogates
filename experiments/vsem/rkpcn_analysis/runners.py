# experiments/vsem/rkpcn_analysis/runners.py
"""
Run RKPCN v2 variants with full trace output for diagnostic analysis.

Provides:
- ``run_rkpcn_variant``: Run a single RKPCN chain using the v2 kernel,
  returning the full (unthinned) trace with diagnostics.
- ``get_adapted_proposal``: Short exact-posterior MCMC to get a
  well-tuned proposal covariance.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from uncprop.core.rkpcn import (
    RKPCNConfig,
    build_rkpcn_kernel,
    build_log_density_vsem,
)
from uncprop.core.rkpcn_adaptation import (
    AdaptiveRKPCNConfig,
    build_adaptive_rkpcn_kernel,
    get_adapted_proposal_cov,
)
from uncprop.core.samplers import mcmc_loop, sample_distribution
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


def get_adapted_proposal(key, rep, n_warmup=5000, n_burnin=5000):
    """Run a short exact-posterior MCMC to get an adapted proposal covariance.

    Uses the existing adaptive MH sampler targeting the exact posterior
    to obtain a reasonable u-proposal covariance for RKPCN.

    Args:
        key: PRNG key.
        rep: VSEMReplicate with posterior attribute.
        n_warmup: Number of samples after burn-in (for adaptation).
        n_burnin: Burn-in before collecting adaptation samples.

    Returns:
        prop_cov: (d, d) adapted proposal covariance matrix.
        exact_samples: (n_warmup, d) samples from the exact posterior.
    """
    key_init, key_mcmc = jr.split(key)
    initial_position = rep.posterior.prior.sample(key_init)

    results = sample_distribution(
        key=key_mcmc,
        dist=rep.posterior,
        initial_position=initial_position,
        n_samples=n_warmup,
        n_burnin=n_burnin,
        thin_window=1,
    )

    return results['prop_cov'], results['positions'].squeeze(1)


def run_rkpcn_variant(
    key,
    rep,
    rho: float = 0.99,
    n_u_steps: int = 1,
    prop_cov=None,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    initial_position=None,
    label: str | None = None,
):
    """Run a single RKPCN v2 chain and return the full trace.

    Builds an RKPCNConfig, constructs the kernel, runs via mcmc_loop,
    and returns diagnostics. The full unthinned trace is returned for
    detailed analysis.

    Args:
        key: PRNG key.
        rep: VSEMReplicate with posterior, posterior_surrogate attributes.
        rho: pCN correlation parameter.
        n_u_steps: Number of u-updates per f-update.
        prop_cov: (d, d) proposal covariance for u-updates.
            Can also be a scalar (interpreted as scale * I) or None
            (uses 0.01 * I).
        n_total: Total number of MCMC macro-iterations (burnin + post).
        n_burnin: Number of burn-in iterations.
        initial_position: Starting position. If None, sampled from prior.
        label: Optional display label.

    Returns:
        dict with keys:
            positions: (n_total, d) — u positions at every iteration
            logdensities: (n_total,) — log-density at every iteration
            accept_probs: (n_total,) — MH acceptance probability
            is_accepted: (n_total,) — whether u-proposal was accepted
            post_burnin: (n_post, d) — positions after burn-in
            ess: (d,) — ESS per dimension (post-burnin)
            accept_rate: float — mean acceptance rate (post-burnin)
            rho: float
            n_u_steps: int
            label: str
            n_burnin: int
            runtime: float — wall-clock seconds
    """
    key_init, key_samp = jr.split(key)

    surr = rep.posterior_surrogate
    log_density_fn = build_log_density_vsem(rep.posterior, surr)
    gp = surr.surrogate

    d = rep.posterior.dim

    # Handle prop_cov: None → default, scalar → scale*I
    if prop_cov is None:
        prop_cov = 0.01 * jnp.eye(d)
    elif jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    # Initial position
    if initial_position is None:
        initial_position = rep.posterior.prior.sample(key_init)
    initial_position = jnp.atleast_1d(jnp.squeeze(initial_position))

    # Build v2 kernel
    config = RKPCNConfig(rho=rho, n_u_steps=n_u_steps)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)
    state = init_fn(key_init, initial_position, prop_cov)

    if label is None:
        parts = [f'rho{int(rho*100)}']
        if n_u_steps > 1:
            parts.append(f'u{n_u_steps}')
        label = '_'.join(parts)

    print(f'  Running {label} (rho={rho}, n_u_steps={n_u_steps}, '
          f'n_total={n_total}, n_burnin={n_burnin})...')

    start_time = time.perf_counter()
    states, infos = mcmc_loop(key=key_samp, kernel=kernel_fn,
                              initial_state=state, num_samples=n_total)
    runtime = time.perf_counter() - start_time

    positions = np.array(states.position)
    logdensities = np.array(states.logdensity)
    accept_probs = np.array(infos.accept_prob)
    is_accepted = np.array(infos.is_accepted)

    post_burnin = positions[n_burnin:]
    ess = compute_ess(post_burnin)
    accept_rate = float(np.mean(accept_probs[n_burnin:]))

    print(f'    accept={accept_rate:.4f}, min_ESS={min(ess):.1f}, '
          f'n_post={post_burnin.shape[0]}, time={runtime:.1f}s')

    return {
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
    }


def run_rkpcn_adaptive_variant(
    key,
    rep,
    rho: float = 0.99,
    n_u_steps: int = 1,
    prop_cov=None,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    adapt_end: int | None = None,
    adapt_interval: int = 50,
    target_accept: float = 0.234,
    gamma_exponent: float = 0.8,
    initial_position=None,
    label: str | None = None,
):
    """Run an adaptive RKPCN chain and return the full trace.

    Uses the adaptive kernel that tunes the u-proposal covariance
    during the first ``adapt_end`` iterations (defaults to ``n_burnin``),
    then freezes for the remainder.

    Args:
        key: PRNG key.
        rep: VSEMReplicate.
        rho: pCN correlation parameter.
        n_u_steps: Number of u-updates per f-update.
        prop_cov: Initial proposal covariance (adapted from this starting point).
        n_total: Total iterations.
        n_burnin: Burn-in iterations (for post-processing, not adaptation).
        adapt_end: Stop adapting after this many iterations.
            Defaults to ``n_burnin`` if not set.
        adapt_interval: Steps between adaptation updates.
        target_accept: Target acceptance rate for Robbins-Monro.
        gamma_exponent: Decay rate for adaptation step size.
        initial_position: Starting position.
        label: Display label.

    Returns:
        Same dict as ``run_rkpcn_variant``, plus:
            adapted_prop_cov_diag: final adapted proposal covariance diagonal.
            adaptive: True (flag for downstream code).
    """
    key_init, key_samp = jr.split(key)

    surr = rep.posterior_surrogate
    log_density_fn = build_log_density_vsem(rep.posterior, surr)
    gp = surr.surrogate

    d = rep.posterior.dim

    if prop_cov is None:
        prop_cov = 0.01 * jnp.eye(d)
    elif jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    if initial_position is None:
        initial_position = rep.posterior.prior.sample(key_init)
    initial_position = jnp.atleast_1d(jnp.squeeze(initial_position))

    if adapt_end is None:
        adapt_end = n_burnin

    config = RKPCNConfig(rho=rho, n_u_steps=n_u_steps)
    adapt_config = AdaptiveRKPCNConfig(
        adapt_end=adapt_end,
        adapt_interval=adapt_interval,
        target_accept=target_accept,
        gamma_exponent=gamma_exponent,
    )

    init_fn, kernel_fn = build_adaptive_rkpcn_kernel(
        config, adapt_config, log_density_fn, gp)
    state = init_fn(key_init, initial_position, prop_cov)

    if label is None:
        parts = [f'rho{int(rho*100)}', 'adapt']
        if n_u_steps > 1:
            parts.append(f'u{n_u_steps}')
        label = '_'.join(parts)

    print(f'  Running {label} (rho={rho}, n_u_steps={n_u_steps}, '
          f'adaptive, adapt_end={adapt_end}, '
          f'n_total={n_total}, n_burnin={n_burnin})...')

    start_time = time.perf_counter()
    states, infos = mcmc_loop(key=key_samp, kernel=kernel_fn,
                              initial_state=state, num_samples=n_total)
    runtime = time.perf_counter() - start_time

    positions = np.array(states.position)
    logdensities = np.array(states.logdensity)
    accept_probs = np.array(infos.accept_prob)
    is_accepted = np.array(infos.is_accepted)

    post_burnin = positions[n_burnin:]
    ess = compute_ess(post_burnin)
    accept_rate = float(np.mean(accept_probs[n_burnin:]))

    # Extract final adapted proposal covariance
    final_tril = np.array(states.proposal_tril[-1])
    adapted_cov = final_tril @ final_tril.T

    print(f'    accept={accept_rate:.4f}, min_ESS={min(ess):.1f}, '
          f'n_post={post_burnin.shape[0]}, time={runtime:.1f}s')
    print(f'    adapted prop_cov diag: {np.diag(adapted_cov)}')

    return {
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
        'adapted_prop_cov_diag': np.diag(adapted_cov).tolist(),
        'adaptive': True,
    }


def run_rkpcn_multi_chain(
    key,
    rep,
    n_chains: int = 4,
    rho: float = 0.99,
    n_u_steps: int = 1,
    prop_cov=None,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    adaptive: bool = True,
    adapt_interval: int = 50,
    target_accept: float = 0.234,
    init_method: str = 'ep_direct_sampling',
    init_n_candidates: int = 500,
    init_n_trials: int = 100,
    weight_method: str = 'pritchard',
    label: str | None = None,
):
    """Run multiple RKPCN chains from diverse starting positions.

    Combines the chains using importance-like weights (Pritchard or
    other method), detects failed chains, and identifies duplicate modes.

    Parameters
    ----------
    key : PRNG key
    rep : VSEMReplicate
    n_chains : int
        Number of chains to run.
    rho : float
        pCN correlation parameter (shared across chains).
    n_u_steps : int
        u-steps per f-update (shared across chains).
    prop_cov : initial proposal covariance. If None, uses 0.01*I.
    n_total : total iterations per chain.
    n_burnin : burn-in per chain.
    adaptive : whether to use adaptive proposal.
    adapt_interval : adaptation batch size.
    target_accept : target acceptance rate for adaptation.
    init_method : initial position selection strategy.
    weight_method : chain weighting method ('equal', 'mean_logdens', 'pritchard').
    label : display label.

    Returns
    -------
    dict with standard result fields (post_burnin = pooled samples), plus:
        n_chains, chain_weights, per_chain_results, failed_mask,
        mode_labels, init_positions, sample_weights
    """
    key_init, key_chains = jr.split(key)

    surr = rep.posterior_surrogate
    log_density_fn = build_log_density_vsem(rep.posterior, surr)
    gp = surr.surrogate
    d = rep.posterior.dim

    if prop_cov is None:
        prop_cov = 0.01 * jnp.eye(d)
    elif jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    if label is None:
        label = f'rho{int(rho*100)}_{n_chains}ch'

    # Select diverse starting positions.
    # The EP-aware default samples approximately from the expected
    # posterior using joint GP trajectories at candidate points. This
    # requires the surrogate posterior's log-density-from-samples
    # interface (implemented by both log-density and forward-model
    # surrogates). We pass ``surrogate_post`` (the SurrogateDistribution)
    # rather than the prior: chains target the surrogate-induced EP,
    # and the surrogate's support may be strictly narrower than the
    # prior's (unbounded priors would leave chains stuck outside).
    print(f'  Selecting {n_chains} initial positions (method={init_method})...')
    init_positions = select_initial_positions(
        key_init,
        surrogate_post=surr,
        n_chains=n_chains,
        method=init_method,
        n_candidates=init_n_candidates,
        n_trials=init_n_trials,
    )
    init_positions = np.array(init_positions)
    print(f'    positions: {init_positions}')

    # Build kernel factory
    config = RKPCNConfig(rho=rho, n_u_steps=n_u_steps)

    if adaptive:
        adapt_config = AdaptiveRKPCNConfig(
            adapt_end=n_burnin,
            adapt_interval=adapt_interval,
            target_accept=target_accept,
        )
        def build_kernel_fn():
            return build_adaptive_rkpcn_kernel(
                config, adapt_config, log_density_fn, gp)
    else:
        def build_kernel_fn():
            return build_rkpcn_kernel(config, log_density_fn, gp)

    # Run all chains
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

    # Detect failures
    failed_mask, fail_diag = detect_failed_chains(chain_results)

    # Identify modes via pairwise cross-chain R-hat (failed chains excluded).
    # After this, mode_labels[m] is -1 for failed chains or a cluster index
    # >= 0 for chains merged into the same distribution.
    mode_labels = identify_duplicate_modes(
        chain_results, failed_mask=failed_mask)

    # Merge chains within each mode into a single pooled sample set.
    # mode_results is a list of K dicts (K = number of distinct non-failed
    # modes), each containing the concatenated post_burnin + logdensities
    # across member chains.
    mode_results = merge_chains_by_mode(chain_results, mode_labels)

    # Compute weights AT THE MODE LEVEL — one weight per mode, computed
    # from the pooled samples within that mode. This avoids the pathological
    # per-chain weight concentration we saw when chains in the same mode
    # had similar but non-identical log-density statistics.
    mode_weights = compute_chain_weights(
        mode_results, method=weight_method, n_burnin=0)

    # Print summary (per-chain diagnostics + per-mode summary)
    par_names = list(rep.grid.dim_names) if hasattr(rep, 'grid') else None
    print_multi_chain_summary(
        chain_results, mode_results=mode_results,
        mode_weights=mode_weights,
        failed_mask=failed_mask, labels=mode_labels, par_names=par_names)

    # Combine modes (samples equally weighted within each mode)
    if len(mode_results) > 0:
        pooled_samples, sample_weights = combine_chains(
            mode_results, mode_weights, n_burnin=0)
    else:
        # No non-failed chains
        pooled_samples = np.empty((0, d))
        sample_weights = np.empty(0)

    # Aggregate diagnostics
    total_runtime = sum(r['runtime'] for r in chain_results)
    pooled_ess = (compute_ess(pooled_samples)
                  if pooled_samples.shape[0] > 10 else np.zeros(d))

    # Weighted mean acceptance rate: use per-chain weights derived from
    # the mode weights (each chain gets its mode's weight divided equally
    # among chains in that mode).
    per_chain_weights = np.zeros(n_chains)
    for k, mr in enumerate(mode_results):
        n_in_mode = len(mr['chain_indices'])
        if n_in_mode > 0:
            for c in mr['chain_indices']:
                per_chain_weights[c] = mode_weights[k] / n_in_mode

    if per_chain_weights.sum() > 0:
        mean_accept = float(np.average(
            [r['accept_rate'] for r in chain_results],
            weights=per_chain_weights))
    else:
        mean_accept = 0.0

    # Build logdensity array by concatenating each mode's logdensities
    # (already trimmed in detect_failed_chains)
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
        'post_burnin': pooled_samples,
        'sample_weights': sample_weights,
        'logdensities': all_ld,
        'positions': pooled_samples,  # for multi-chain, positions = pooled post-burnin
        'accept_probs': np.concatenate([r['accept_probs'][n_burnin:] for r in chain_results]),
        'is_accepted': np.concatenate([
            r['accept_probs'][n_burnin:] > 0.5 for r in chain_results]),
        'ess': pooled_ess,
        'accept_rate': mean_accept,
        'rho': rho,
        'n_u_steps': n_u_steps,
        'label': label,
        'n_burnin': 0,  # post_burnin and logdensities are already trimmed
        'runtime': total_runtime,
        # Multi-chain specific
        'n_chains': n_chains,
        'mode_weights': mode_weights,
        'mode_results': mode_results,
        'mode_membership': [mr['chain_indices'] for mr in mode_results],
        'per_chain_results': chain_results,
        'failed_mask': failed_mask,
        'mode_labels': mode_labels,
        'init_positions': init_positions,
        'weight_method': weight_method,
        'multi_chain': True,
    }
