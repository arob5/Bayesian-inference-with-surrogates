# experiments/vsem/rkpcn_analysis/runners.py
"""
Run RKPCN variants with full trace output for diagnostic analysis.

The main entry point is `run_rkpcn_variant`, which runs a single RKPCN chain
and returns the full (unthinned) MCMC trace: positions, log-densities,
acceptance probabilities, and f-values at every iteration.

`run_rkpcn_sweep` runs multiple variants (rho, proposal scale, etc.) and
returns a dict of results keyed by variant label.
"""

from typing import NamedTuple
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from uncprop.core.samplers import (
    mcmc_loop,
    init_rkpcn_kernel,
    _f_update_pcn_proposal,
    _f_update_eup_cpm,
    sample_distribution,
)
from uncprop.core.inverse_problem import Posterior
from uncprop.models.vsem.surrogate import (
    VSEMPosteriorSurrogate,
    LogDensClippedGPSurrogate,
)
from uncprop.utils.diagnostics import compute_ess


def build_log_density(posterior: Posterior,
                      surrogate_post: VSEMPosteriorSurrogate):
    """Build the log-density function used by RKPCN.

    For clipped GP surrogates, clips the GP output at the deterministic
    upper bound (prior + likelihood bound). Enforces prior support.

    Args:
        posterior: The exact posterior object.
        surrogate_post: The surrogate posterior object.

    Returns:
        Callable (f, u) -> scalar log-density.
    """
    low, high = surrogate_post.support
    upper_bound = (
        lambda u: posterior.prior.log_density(u)
              + posterior.likelihood.log_density_upper_bound(u)
    )

    if isinstance(surrogate_post, LogDensClippedGPSurrogate):
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            upper = upper_bound(u)
            lp = jnp.clip(f, max=upper)
            lp = jnp.where(
                jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()
    else:
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            lp = f
            lp = jnp.where(
                jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()

    return log_density


def get_adapted_proposal(key, rep, n_warmup=5000, n_burnin=5000):
    """Run a short exact-posterior MCMC to get an adapted proposal covariance.

    This provides a reasonable u-proposal for RKPCN by adapting to the
    exact posterior geometry.

    Args:
        key: PRNG key.
        rep: VSEMReplicate with posterior attribute.
        n_warmup: Number of adaptation samples.
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
    rho: float,
    prop_cov,
    n_total: int = 55_000,
    n_burnin: int = 50_000,
    initial_position=None,
    f_update_fn=None,
    label: str | None = None,
):
    """Run a single RKPCN chain and return the full trace.

    Args:
        key: PRNG key.
        rep: VSEMReplicate with posterior, posterior_surrogate attributes.
        rho: pCN correlation parameter.
        prop_cov: (d, d) proposal covariance for u-updates.
            Can also be a scalar, interpreted as `scale * I`.
        n_total: Total number of MCMC iterations (burnin + post-burnin).
        n_burnin: Number of burn-in iterations.
        initial_position: Starting position. If None, sampled from prior.
        f_update_fn: The f-update function. Defaults to _f_update_pcn_proposal
            (no MH correction on f). Use _f_update_eup_cpm for EUP targeting.
        label: Optional label for this variant (for display).

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
            label: str
    """
    if f_update_fn is None:
        f_update_fn = _f_update_pcn_proposal

    key_ker, key_init, key_samp = jr.split(key, 3)

    surr = rep.posterior_surrogate
    log_density = build_log_density(rep.posterior, surr)
    gp = surr.surrogate

    # Handle scalar prop_cov
    d = rep.posterior.dim
    if jnp.ndim(prop_cov) == 0:
        prop_cov = float(prop_cov) * jnp.eye(d)

    # Initial position
    if initial_position is None:
        initial_position = rep.posterior.prior.sample(key_init)
    initial_position = jnp.squeeze(initial_position)

    # f-update info
    class UpdateInfo(NamedTuple):
        rho: float
    f_update_info = UpdateInfo(rho=rho)

    init_fn, kernel = init_rkpcn_kernel(
        key=key_ker, log_density=log_density, gp=gp,
        f_update_fn=f_update_fn, f_update_info=f_update_info)
    initial_state = init_fn(key_ker, initial_position, prop_cov)

    if label is None:
        label = f'rkpcn{int(rho*100)}'
    print(f'  Running {label} (rho={rho}, n_total={n_total}, '
          f'n_burnin={n_burnin})...')

    states, infos = mcmc_loop(
        key=key_samp, kernel=kernel,
        initial_state=initial_state, num_samples=n_total)

    positions = np.array(states.position)
    logdensities = np.array(states.logdensity)
    accept_probs = np.array(infos.accept_prob)
    is_accepted = np.array(infos.is_accepted)

    post_burnin = positions[n_burnin:]
    ess = compute_ess(post_burnin)
    accept_rate = float(np.mean(accept_probs[n_burnin:]))

    print(f'    accept_rate={accept_rate:.4f}, '
          f'min_ESS={min(ess):.1f}, '
          f'n_post={post_burnin.shape[0]}')

    return {
        'positions': positions,
        'logdensities': logdensities,
        'accept_probs': accept_probs,
        'is_accepted': is_accepted,
        'post_burnin': post_burnin,
        'ess': ess,
        'accept_rate': accept_rate,
        'rho': rho,
        'label': label,
        'n_burnin': n_burnin,
    }


def run_rkpcn_sweep(key, rep, variants, prop_cov, **common_kwargs):
    """Run multiple RKPCN variants and collect results.

    Args:
        key: PRNG key.
        rep: VSEMReplicate.
        variants: list of dicts, each with at least 'rho' and optionally
            'label', 'prop_cov', 'n_total', 'n_burnin', 'f_update_fn'.
            Fields in the variant dict override common_kwargs.
        prop_cov: Default proposal covariance (overridden per-variant if set).
        **common_kwargs: Passed to run_rkpcn_variant for all variants.

    Returns:
        dict mapping label -> result dict.
    """
    results = {}

    for i, variant in enumerate(variants):
        jax.clear_caches()
        key, subkey = jr.split(key)

        kwargs = dict(common_kwargs)
        kwargs.update(variant)
        v_prop_cov = kwargs.pop('prop_cov', prop_cov)

        result = run_rkpcn_variant(
            key=subkey, rep=rep, prop_cov=v_prop_cov, **kwargs)
        results[result['label']] = result

    return results
