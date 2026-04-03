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
from uncprop.core.samplers import mcmc_loop, sample_distribution
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
