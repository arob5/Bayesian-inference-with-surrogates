# uncprop/core/rkpcn.py
"""
RKPCN: Modular Random Kernel preconditioned Crank-Nicolson sampler.

This module provides a clean, blackjax-compatible RKPCN implementation
designed for easy experimentation with algorithm variants. The kernel is
compatible with ``mcmc_loop`` and ``mcmc_loop_multiple_chains`` from
``uncprop.core.samplers``.

Algorithm overview
------------------
RKPCN targets the expected posterior (EP)::

    pi_hat(u) = E_f[pi(u; f)]  where  f ~ GP(mu, k)

by maintaining an extended state (u, f(u)) and alternating:

1. **f-update** (pCN proposal at u, no MH correction):
   Apply pCN to get g(u) from f(u). Condition the GP on {u, g(u)}.

2. **u-update(s)** (MH with proposal covariance Sigma_Q):
   For each u-step: propose v, sample g(v) from conditioned GP,
   accept/reject via pi(v; g(v)) / pi(u; g(u)), then condition
   GP on {v, g(v)} for subsequent steps.

The f-step and u-steps are cleanly separated. The f-step only instantiates
the trajectory at the current position u (and later, support points).
The u-steps iteratively condition the GP as they explore.

Designed for extension with:
- Adaptive proposal covariance (Phase 3)
- Bridging strategies between f and g (Phase 5)
- Support points for normalizing constant estimation (Phase 6)

Usage
-----
::

    from uncprop.core.rkpcn import RKPCNConfig, build_rkpcn_kernel
    from uncprop.core.samplers import mcmc_loop

    config = RKPCNConfig(rho=0.99, n_u_steps=1)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)
    state = init_fn(key, initial_position, prop_cov)
    states, infos = mcmc_loop(key, kernel_fn, state, num_samples=10_000)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array, ArrayLike
from uncprop.core.surrogate import GPJaxSurrogate
from uncprop.utils.distribution import (
    _sample_gaussian_tril,
    _sample_batch_gaussian_tril,
)


# =============================================================================
# State and info types
# =============================================================================

class RKPCNState(NamedTuple):
    """State of the RKPCN sampler.

    Attributes
    ----------
    position : Array, shape (d,)
        Current parameter position u.
    f_position : Array, shape (q,)
        Current GP trajectory value g(u), where q is the GP output dim.
    logdensity : float
        Log-density log pi_tilde(u; g(u)) at the current state.
    proposal_tril : Array, shape (d, d)
        Lower Cholesky factor of the u-proposal covariance Sigma_Q.
    """
    position: Array
    f_position: Array
    logdensity: Array
    proposal_tril: Array


class RKPCNInfo(NamedTuple):
    """Per-iteration diagnostics from the RKPCN kernel.

    Attributes
    ----------
    accept_prob : float
        MH acceptance probability for the u-update. When n_u_steps > 1,
        this is the mean acceptance probability across all u-steps.
    is_accepted : bool
        Whether any u-proposal was accepted in this macro-iteration.
    """
    accept_prob: ArrayLike
    is_accepted: ArrayLike


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class RKPCNConfig:
    """Static configuration for the RKPCN kernel.

    Attributes
    ----------
    rho : float
        pCN correlation parameter in [0, 1). Controls the magnitude of
        the f-update perturbation. Higher rho means smaller steps in
        trajectory space.
    n_u_steps : int
        Number of u-updates per f-update. More u-steps give the
        parameter chain time to equilibrate under the current trajectory
        before the next f-update. Default 1.
    """
    rho: float = 0.99
    n_u_steps: int = 1


# =============================================================================
# Kernel builder
# =============================================================================

def build_rkpcn_kernel(
    config: RKPCNConfig,
    log_density_fn: Callable[[Array, Array], Array],
    gp: GPJaxSurrogate,
) -> tuple[Callable, Callable]:
    """Build an RKPCN kernel compatible with ``mcmc_loop``.

    Parameters
    ----------
    config : RKPCNConfig
        Static configuration (rho, n_u_steps, etc.).
    log_density_fn : callable
        Function ``(f, u) -> scalar`` computing the log unnormalized
        density. ``f`` has shape ``(q,)`` (GP output at u) and ``u``
        has shape ``(d,)``. The argument ``u`` is passed for use in
        enforcing support constraints.
    gp : GPJaxSurrogate
        The GP surrogate model. Must support ``gp(u)`` for prediction,
        ``gp.condition(given=(u, f))`` for conditioning, and
        ``gp.condition_then_predict(u_new, given=(u, f))`` for
        combined conditioning + prediction.

    Returns
    -------
    init_fn : callable
        ``(key, position, prop_cov) -> RKPCNState``
    kernel_fn : callable
        ``(key, state) -> (RKPCNState, RKPCNInfo)``
        Compatible with ``mcmc_loop`` and ``mcmc_loop_multiple_chains``.
    """
    rho = config.rho
    n_u_steps = config.n_u_steps

    # ------------------------------------------------------------------
    # Kernel function (one macro-iteration)
    # ------------------------------------------------------------------
    def kernel_fn(key: PRNGKey, state: RKPCNState) -> tuple[RKPCNState, RKPCNInfo]:
        key_f, key_u = jr.split(key)

        # --- f-update: univariate pCN at u, condition GP ---
        state, gp_conditioned = _f_update_pcn(key_f, state, gp, log_density_fn, rho)

        # --- u-update(s): MH steps using conditioned GP ---
        state, info = _u_updates(key_u, state, gp_conditioned, log_density_fn, n_u_steps)

        return state, info

    # ------------------------------------------------------------------
    # Init function
    # ------------------------------------------------------------------
    def init_fn(key: PRNGKey, position: Array, prop_cov: Array) -> RKPCNState:
        """Initialize RKPCN state.

        Samples f(u_0) from the GP marginal and evaluates the log-density.

        Parameters
        ----------
        key : PRNGKey
        position : Array, shape (d,)
            Initial parameter position.
        prop_cov : Array, shape (d, d)
            Initial u-proposal covariance matrix.
        """
        position = jnp.atleast_1d(jnp.squeeze(position))
        prop_tril = jnp.linalg.cholesky(prop_cov, upper=False)

        f_init = gp(position).sample(key).reshape(gp.output_dim)
        lp_init = log_density_fn(f_init, position).squeeze()

        return RKPCNState(
            position=position,
            f_position=f_init,
            logdensity=lp_init,
            proposal_tril=prop_tril,
        )

    return init_fn, kernel_fn


# =============================================================================
# f-update: univariate pCN at u
# =============================================================================

def _f_update_pcn(
    key: PRNGKey,
    state: RKPCNState,
    gp: GPJaxSurrogate,
    log_density_fn: Callable,
    rho: float,
) -> tuple[RKPCNState, GPJaxSurrogate]:
    """Apply a pCN f-update at the current position u.

    Draws a new trajectory value g(u) via the pCN proposal applied to
    the current value f(u). Does NOT apply a Metropolis correction
    (the approximation alpha_f = 1). After the update, conditions the
    GP on {u, g(u)} to prepare for subsequent u-updates.

    Parameters
    ----------
    key : PRNGKey
    state : RKPCNState
        Current state with position u and f_position = f(u).
    gp : GPJaxSurrogate
        The (unconditioned) GP surrogate.
    log_density_fn : callable
        (f, u) -> scalar log-density.
    rho : float
        pCN correlation parameter.

    Returns
    -------
    state : RKPCNState
        Updated state with f_position = g(u) and updated logdensity.
    gp_conditioned : GPJaxSurrogate
        GP conditioned on {u, g(u)}, ready for u-step predictions.
    """
    u = state.position
    fu = state.f_position

    # Univariate pCN: g(u) from f(u)
    # Get the GP marginal at u for the pCN mean and variance
    gp_at_u = gp(u)
    gu = _pcn_proposal_univariate(key, fu, gp_at_u.mean, gp_at_u.chol, rho)

    # Update state
    new_logdensity = log_density_fn(gu, u)
    state = state._replace(
        f_position=gu,
        logdensity=new_logdensity,
    )

    # Condition GP on {u, g(u)} for subsequent u-updates
    gp_conditioned = gp.condition(given=(u, gu))

    return state, gp_conditioned


# =============================================================================
# u-update(s): MH steps with iterative GP conditioning
# =============================================================================

def _u_updates(
    key: PRNGKey,
    state: RKPCNState,
    gp_conditioned: GPJaxSurrogate,
    log_density_fn: Callable,
    n_steps: int,
) -> tuple[RKPCNState, RKPCNInfo]:
    """Perform one or more MH u-updates under the current trajectory g.

    At each step:
    1. Propose v ~ N(u_current, Sigma_Q)
    2. Sample g(v) from the conditioned GP
    3. Condition GP on {v, g(v)} (for subsequent steps)
    4. MH accept/reject using pi(v; g(v)) / pi(u; g(u))
    5. Update position if accepted

    The conditioning set grows with each step, giving better
    predictions for subsequent proposals.

    Parameters
    ----------
    key : PRNGKey
    state : RKPCNState
    gp_conditioned : GPJaxSurrogate
        GP already conditioned on {u, g(u)} from the f-step.
    log_density_fn : callable
    n_steps : int
        Number of u-updates to perform.

    Returns
    -------
    state : RKPCNState
        Updated state after all u-steps.
    info : RKPCNInfo
        accept_prob: mean acceptance probability across all steps.
        is_accepted: whether any step was accepted.
    """
    if n_steps == 1:
        # Fast path: single u-step, no scan overhead
        return _single_u_step(key, state, gp_conditioned, log_density_fn)

    # Multiple u-steps: use Python loop (not jax.lax.scan) because
    # each step produces a differently-sized conditioned GP (precision
    # matrix grows by one row/column). With n_u_steps typically small
    # (1-10), the unrolled loop overhead is minimal.
    #
    # Note: since n_u_steps is a compile-time constant (from config),
    # this loop is unrolled at JIT trace time, producing a fixed
    # computation graph. JAX values stay as traced arrays throughout.
    total_accept_prob = jnp.array(0.0)
    any_accepted = jnp.array(False)

    keys = jr.split(key, n_steps)
    for i in range(n_steps):
        state, gp_conditioned, step_info = _single_u_step_and_condition(
            keys[i], state, gp_conditioned, log_density_fn)
        total_accept_prob = total_accept_prob + step_info.accept_prob
        any_accepted = any_accepted | step_info.is_accepted

    mean_accept = total_accept_prob / n_steps
    info = RKPCNInfo(
        accept_prob=mean_accept,
        is_accepted=any_accepted,
    )

    return state, info


def _single_u_step(
    key: PRNGKey,
    state: RKPCNState,
    gp_conditioned: GPJaxSurrogate,
    log_density_fn: Callable,
) -> tuple[RKPCNState, RKPCNInfo]:
    """Single MH u-step. Does not further condition the GP (no next step).

    Parameters
    ----------
    key : PRNGKey
    state : RKPCNState
    gp_conditioned : GPJaxSurrogate
        GP conditioned on all previously observed trajectory values.
    log_density_fn : callable

    Returns
    -------
    state : RKPCNState
    info : RKPCNInfo
    """
    key_prop, key_sample, key_mh = jr.split(key, 3)

    u = state.position

    # Propose v ~ N(u, Sigma_Q)
    v = _sample_gaussian_tril(key_prop, m=u, L=state.proposal_tril).squeeze()

    # Sample g(v) from GP conditioned on all observed points
    gv = _gp_sample_at(key_sample, gp_conditioned, v)

    # MH accept/reject
    lp_prop = log_density_fn(gv, v)
    u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
        key_mh, state.logdensity, lp_prop, u, v)

    # Update f_position to track the accepted position's trajectory value
    f_next = jax.lax.cond(
        is_accepted,
        lambda _: gv,
        lambda _: state.f_position,
        operand=None,
    )

    new_state = state._replace(
        position=u_next,
        f_position=f_next,
        logdensity=lp_next,
    )

    return new_state, RKPCNInfo(accept_prob=accept_prob, is_accepted=is_accepted)


def _single_u_step_and_condition(
    key: PRNGKey,
    state: RKPCNState,
    gp_conditioned: GPJaxSurrogate,
    log_density_fn: Callable,
) -> tuple[RKPCNState, GPJaxSurrogate, RKPCNInfo]:
    """Single MH u-step, then condition GP on the evaluated point.

    Used in the multi-u-step loop. After evaluating g(v), conditions
    the GP on {v, g(v)} regardless of acceptance (the trajectory value
    is informative either way).

    Returns
    -------
    state : RKPCNState
    gp_conditioned : GPJaxSurrogate
        Further conditioned GP.
    info : RKPCNInfo
    """
    key_prop, key_sample, key_mh = jr.split(key, 3)

    u = state.position

    # Propose v ~ N(u, Sigma_Q)
    v = _sample_gaussian_tril(key_prop, m=u, L=state.proposal_tril).squeeze()

    # Sample g(v) from conditioned GP
    gv = _gp_sample_at(key_sample, gp_conditioned, v)

    # Condition GP on {v, g(v)} for subsequent steps
    gp_conditioned = gp_conditioned.condition(given=(v, gv))

    # MH accept/reject
    lp_prop = log_density_fn(gv, v)
    u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
        key_mh, state.logdensity, lp_prop, u, v)

    f_next = jax.lax.cond(
        is_accepted,
        lambda _: gv,
        lambda _: state.f_position,
        operand=None,
    )

    new_state = state._replace(
        position=u_next,
        f_position=f_next,
        logdensity=lp_next,
    )

    return new_state, gp_conditioned, RKPCNInfo(accept_prob=accept_prob, is_accepted=is_accepted)


# =============================================================================
# GP helpers
# =============================================================================

def _gp_sample_at(
    key: PRNGKey,
    gp: GPJaxSurrogate,
    u: Array,
) -> Array:
    """Sample g(u) from GP (conditioned or unconditioned) at a single point.

    Parameters
    ----------
    key : PRNGKey
    gp : GPJaxSurrogate
    u : Array, shape (d,)

    Returns
    -------
    f_u : Array, shape (q,)
    """
    pred = gp(u)
    return pred.sample(key).squeeze(0).reshape(gp.output_dim)


def _pcn_proposal_univariate(
    key: PRNGKey,
    f_u: Array,
    mean: Array,
    cov_tril: Array,
    rho: float,
) -> Array:
    """Univariate pCN proposal at a single point.

    Applies the pCN formula to the current trajectory value f(u) using
    the GP marginal distribution at u.

    Parameters
    ----------
    key : PRNGKey
    f_u : Array, shape (q,)
        Current trajectory value at u.
    mean : Array, shape (q,) or (q, 1)
        GP prior mean at u.
    cov_tril : Array, shape (q, 1, 1) or similar
        GP prior Cholesky at u.
    rho : float
        pCN correlation.

    Returns
    -------
    g_u : Array, shape (q,)
        Proposed new trajectory value at u.
    """
    # Flatten to (q,) for scalar operations
    mean = mean.ravel()
    f_u = f_u.ravel()
    q = f_u.shape[0]

    # pCN: g(u) = mu + rho*(f(u) - mu) + sqrt(1-rho^2)*noise
    pcn_mean = mean + rho * (f_u - mean)

    # Extract marginal std from Cholesky (diagonal of 1x1 Cholesky per output)
    # cov_tril shape is (q, 1, 1) for a single point
    std = jnp.abs(cov_tril.ravel())  # (q,)
    pcn_std = jnp.sqrt(1 - rho**2) * std

    noise = jr.normal(key, shape=(q,))
    g_u = pcn_mean + pcn_std * noise

    return g_u


# =============================================================================
# MH accept/reject
# =============================================================================

def _mh_accept_reject(
    key: PRNGKey,
    lp_curr: float,
    lp_prop: float,
    u_curr: Array,
    u_prop: Array,
    log_correction: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    """Standard MH accept/reject for symmetric proposals.

    Handles NaN log-densities by treating them as -inf (auto-reject).

    Returns
    -------
    u_next : Array
    lp_next : float
    accept_prob : float, in [0, 1]
    is_accepted : bool
    """
    log_unif = jnp.log(jr.uniform(key))
    log_alpha = jnp.squeeze(lp_prop - lp_curr + log_correction)
    log_alpha = jnp.where(jnp.isnan(log_alpha), -jnp.inf, log_alpha)
    is_accepted = log_unif < log_alpha

    u_next, lp_next = jax.lax.cond(
        is_accepted,
        lambda _: (u_prop, lp_prop),
        lambda _: (u_curr, lp_curr),
        operand=None,
    )

    accept_prob = jnp.clip(jnp.exp(log_alpha), min=0.0, max=1.0)
    return u_next, lp_next, accept_prob, is_accepted


# =============================================================================
# Log-density builders
# =============================================================================

def build_log_density_vsem(posterior, surrogate_post):
    """Build log-density function for VSEM log-posterior emulation.

    Handles both clipped and unclipped GP surrogates, enforcing prior
    support constraints.

    Parameters
    ----------
    posterior : Posterior
        The exact posterior object (for prior and likelihood bounds).
    surrogate_post : VSEMPosteriorSurrogate
        The surrogate posterior object.

    Returns
    -------
    log_density_fn : callable
        ``(f, u) -> scalar`` log-density function compatible with
        ``build_rkpcn_kernel``.
    """
    from uncprop.models.vsem.surrogate import LogDensClippedGPSurrogate

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
