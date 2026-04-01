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


def run_rkpcn_adaptive(
    key,
    rep,
    rho: float,
    prop_cov_init,
    n_burnin: int = 50_000,
    n_post: int = 5_000,
    adapt_start: int = 1000,
    adapt_interval: int = 200,
    target_accept: float = 0.30,
    initial_position=None,
    f_update_fn=None,
    label: str | None = None,
):
    """Run RKPCN with adaptive proposal covariance during burn-in.

    During burn-in, maintains a running sample covariance of accepted
    u-positions and periodically updates the proposal Cholesky factor.
    Also adapts a global scale factor via Robbins-Monro targeting a
    specified acceptance rate.

    After burn-in, the proposal is frozen and the post-burnin phase
    runs via the fast JIT-compiled mcmc_loop.

    The adaptation uses the Haario-style approach:
        Sigma_Q = exp(2*log_s) * (C_hat + eps*I)
    where C_hat is the running sample covariance, log_s is adapted
    to target the desired acceptance rate, and eps is a regularizer.

    Args:
        key: PRNG key.
        rep: VSEMReplicate.
        rho: pCN correlation parameter.
        prop_cov_init: (d, d) initial proposal covariance.
        n_burnin: Number of burn-in iterations (adaptation phase).
        n_post: Number of post-burnin iterations (frozen proposal).
        adapt_start: Begin adapting after this many iterations.
        adapt_interval: Update proposal every this many iterations.
        target_accept: Target acceptance rate for scale adaptation.
        initial_position: Starting position. If None, sampled from prior.
        f_update_fn: The f-update function. Defaults to _f_update_pcn_proposal.
        label: Optional label.

    Returns:
        Same dict as run_rkpcn_variant, plus:
            adapt_history: dict with 'scales', 'accept_rates',
                'cov_traces' at each adaptation step
            final_prop_cov: the frozen proposal covariance after adaptation
    """
    if f_update_fn is None:
        f_update_fn = _f_update_pcn_proposal

    key_ker, key_init, key_burnin, key_post = jr.split(key, 4)

    surr = rep.posterior_surrogate
    log_density = build_log_density(rep.posterior, surr)
    gp = surr.surrogate

    d = rep.posterior.dim
    if jnp.ndim(prop_cov_init) == 0:
        prop_cov_init = float(prop_cov_init) * jnp.eye(d)

    if initial_position is None:
        initial_position = rep.posterior.prior.sample(key_init)
    initial_position = jnp.squeeze(initial_position)

    class UpdateInfo(NamedTuple):
        rho: float
    f_update_info = UpdateInfo(rho=rho)

    init_fn, kernel = init_rkpcn_kernel(
        key=key_ker, log_density=log_density, gp=gp,
        f_update_fn=f_update_fn, f_update_info=f_update_info)
    state = init_fn(key_ker, initial_position, prop_cov_init)

    if label is None:
        label = f'rkpcn{int(rho*100)}_adapt'
    print(f'  Running {label} (rho={rho}, n_burnin={n_burnin}, '
          f'n_post={n_post}, adapt_interval={adapt_interval})...')

    # --- Burn-in with adaptation (Python loop) ---
    eps = 1e-6
    log_scale = 0.0
    gamma_exponent = 0.6  # Robbins-Monro decay rate

    # Running stats (Welford's online algorithm)
    running_mean = np.array(initial_position)
    running_m2 = np.zeros((d, d))  # sum of outer products of deviations
    n_seen = 0

    # Storage for burn-in trace
    burnin_positions = np.zeros((n_burnin, d))
    burnin_logdens = np.zeros(n_burnin)
    burnin_accept = np.zeros(n_burnin)

    # Adaptation history
    adapt_scales = []
    adapt_accept_rates = []
    adapt_cov_diags = []

    # Window for computing rolling acceptance rate
    recent_accepts = []

    burnin_keys = jr.split(key_burnin, n_burnin)

    for t in range(n_burnin):
        state, info = kernel(burnin_keys[t], state)

        pos = np.array(state.position)
        burnin_positions[t] = pos
        burnin_logdens[t] = float(state.logdensity)
        burnin_accept[t] = float(info.accept_prob)
        recent_accepts.append(float(info.is_accepted))

        # Update running covariance (Welford)
        n_seen += 1
        delta = pos - running_mean
        running_mean = running_mean + delta / n_seen
        delta2 = pos - running_mean
        running_m2 = running_m2 + np.outer(delta, delta2)

        # Periodic adaptation
        if t >= adapt_start and (t + 1) % adapt_interval == 0:
            # Current running covariance
            cov_hat = running_m2 / max(n_seen - 1, 1) + eps * np.eye(d)

            # Robbins-Monro scale update
            window_accept = np.mean(recent_accepts[-adapt_interval:])
            n_adapted = len(adapt_scales) + 1
            gamma = 1.0 / (n_adapted ** gamma_exponent)
            log_scale += gamma * (window_accept - target_accept)

            # Build new proposal: scale^2 * cov_hat
            scale = np.exp(log_scale)
            new_prop_cov = scale**2 * cov_hat

            try:
                new_L = np.linalg.cholesky(new_prop_cov)
                state = state._replace(proposal_tril=jnp.array(new_L))
            except np.linalg.LinAlgError:
                pass  # keep old proposal if Cholesky fails

            # Record
            adapt_scales.append(float(scale))
            adapt_accept_rates.append(float(window_accept))
            adapt_cov_diags.append(np.diag(new_prop_cov).copy())

            if (t + 1) % (adapt_interval * 10) == 0:
                print(f'    t={t+1}: scale={scale:.4f}, '
                      f'accept={window_accept:.3f}, '
                      f'cov_diag={np.diag(new_prop_cov)}')

    # Extract final frozen proposal
    final_prop_cov = np.array(state.proposal_tril @ state.proposal_tril.T)
    final_accept = np.mean(burnin_accept[-adapt_interval:])
    print(f'    Burn-in done. Final scale={np.exp(log_scale):.4f}, '
          f'final accept={final_accept:.3f}')
    print(f'    Final prop_cov diag: {np.diag(final_prop_cov)}')

    # --- Post-burnin (JIT-compiled, frozen proposal) ---
    print(f'    Running {n_post} post-burnin iterations (frozen proposal)...')
    post_states, post_infos = mcmc_loop(
        key=key_post, kernel=kernel,
        initial_state=state, num_samples=n_post)

    post_positions = np.array(post_states.position)
    post_logdens = np.array(post_states.logdensity)
    post_accept = np.array(post_infos.accept_prob)
    post_is_accepted = np.array(post_infos.is_accepted)

    # Combine burn-in + post-burnin traces
    all_positions = np.concatenate([burnin_positions, post_positions])
    all_logdens = np.concatenate([burnin_logdens, post_logdens])
    all_accept = np.concatenate([burnin_accept, post_accept])

    # Post-burnin diagnostics
    ess = compute_ess(post_positions)
    accept_rate = float(np.mean(post_accept))

    print(f'    Post-burnin: accept_rate={accept_rate:.4f}, '
          f'min_ESS={min(ess):.1f}')

    return {
        'positions': all_positions,
        'logdensities': all_logdens,
        'accept_probs': all_accept,
        'is_accepted': np.concatenate([
            burnin_accept > np.random.uniform(size=n_burnin),
            post_is_accepted]),
        'post_burnin': post_positions,
        'ess': ess,
        'accept_rate': accept_rate,
        'rho': rho,
        'label': label,
        'n_burnin': n_burnin,
        'final_prop_cov': final_prop_cov,
        'adapt_history': {
            'scales': np.array(adapt_scales),
            'accept_rates': np.array(adapt_accept_rates),
            'cov_diags': np.array(adapt_cov_diags) if adapt_cov_diags else None,
        },
    }


def run_rkpcn_sweep(key, rep, variants, prop_cov, **common_kwargs):
    """Run multiple RKPCN variants and collect results.

    Args:
        key: PRNG key.
        rep: VSEMReplicate.
        variants: list of dicts, each with at least 'rho' and optionally
            'label', 'prop_cov', 'n_total', 'n_burnin', 'f_update_fn'.
            Fields in the variant dict override common_kwargs.
            Set 'adaptive': True to use run_rkpcn_adaptive instead.
            For adaptive runs, use 'prop_cov_init' (or falls back to
            'prop_cov') and 'n_post' instead of 'n_total'.
        prop_cov: Default proposal covariance (overridden per-variant if set).
        **common_kwargs: Passed to the runner function for all variants.

    Returns:
        dict mapping label -> result dict.
    """
    results = {}

    for i, variant in enumerate(variants):
        jax.clear_caches()
        key, subkey = jr.split(key)

        kwargs = dict(common_kwargs)
        kwargs.update(variant)
        is_adaptive = kwargs.pop('adaptive', False)
        v_prop_cov = kwargs.pop('prop_cov', prop_cov)

        if is_adaptive:
            # Adaptive runner uses prop_cov_init, n_post instead of n_total
            kwargs.pop('n_total', None)
            kwargs.setdefault('prop_cov_init', v_prop_cov)
            result = run_rkpcn_adaptive(key=subkey, rep=rep, **kwargs)
        else:
            result = run_rkpcn_variant(
                key=subkey, rep=rep, prop_cov=v_prop_cov, **kwargs)
        results[result['label']] = result

    return results
