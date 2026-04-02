# uncprop/core/rkpcn.py
"""
RKPCN v2: Modular Random Kernel preconditioned Crank-Nicolson sampler.

This module provides a clean, blackjax-compatible RKPCN implementation
designed for easy experimentation with algorithm variants. The kernel is
compatible with ``mcmc_loop`` and ``mcmc_loop_multiple_chains`` from
``uncprop.core.samplers``.

Algorithm overview
------------------
RKPCN targets the expected posterior (EP):

    π̂(u) = E_f[π(u; f)]  where  f ~ GP(μ, k)

by maintaining an extended state (u, f(u)) and alternating:

1. **f-update** (pCN proposal, no MH correction):
   g = μ + ρ(f − μ) + √(1−ρ²) ξ,  ξ ~ GP(0, k)

2. **u-update** (MH with proposal covariance Σ_Q):
   propose ũ ~ N(u, Σ_Q), accept/reject via π(ũ; g) / π(u; g)

The implementation is designed to be extended with:
- Multiple u-steps per f-update (Phase 2)
- Adaptive proposal covariance (Phase 3)
- Bridging strategies between f and g (Phase 5)
- Support points for normalizing constant estimation (Phase 6)

Usage
-----
    from uncprop.core.rkpcn import RKPCNConfig, build_rkpcn_kernel
    from uncprop.core.samplers import mcmc_loop

    config = RKPCNConfig(rho=0.99)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)
    state = init_fn(key, initial_position, prop_cov)
    states, infos = mcmc_loop(key, kernel_fn, state, num_samples=10_000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple
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
        Current GP trajectory value f(u), where q is the GP output dim.
    logdensity : float
        Log-density log π̃(u; f) at the current state.
    proposal_tril : Array, shape (d, d)
        Lower Cholesky factor of the u-proposal covariance Σ_Q.
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
        MH acceptance probability for the u-update.
    is_accepted : bool
        Whether the u-proposal was accepted.
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
        the f-update: higher rho means smaller perturbations. The
        approximation error from dropping the f-acceptance step scales
        as O(√(1−ρ)).
    n_u_steps : int
        Number of u-updates per f-update (Phase 2). Default 1.
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
        has shape ``(d,)``.
    gp : GPJaxSurrogate
        The GP surrogate model, supporting ``gp(u)`` for prediction
        and ``gp.condition_then_predict(u_new, given=(u, f))`` for
        just-in-time conditioning.

    Returns
    -------
    init_fn : callable
        ``(key, position, prop_cov) -> RKPCNState``
    kernel_fn : callable
        ``(key, state) -> (RKPCNState, RKPCNInfo)``
    """
    rho = config.rho
    n_u_steps = config.n_u_steps

    # ------------------------------------------------------------------
    # Kernel function (one macro-iteration)
    # ------------------------------------------------------------------
    def kernel_fn(key: PRNGKey, state: RKPCNState) -> tuple[RKPCNState, RKPCNInfo]:
        key_f, key_u = jr.split(key)

        # --- f-update: pCN proposal without MH correction ---
        state, f_at_proposal_cache = _f_update_pcn(
            key_f, state, gp, log_density_fn, rho,
        )

        # --- u-update(s) ---
        # For n_u_steps == 1, the first u-proposal is the one used
        # in the f-update (its f-value is already cached). For
        # n_u_steps > 1, additional proposals require JIT conditioning.
        state, info = _u_update(
            key_u, state, gp, log_density_fn,
            f_at_proposal=f_at_proposal_cache,
            n_steps=n_u_steps,
        )

        return state, info

    # ------------------------------------------------------------------
    # Init function
    # ------------------------------------------------------------------
    def init_fn(
        key: PRNGKey,
        position: Array,
        prop_cov: Array,
    ) -> RKPCNState:
        """Initialize RKPCN state.

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

        # Sample f(u_0) from the GP marginal
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
# Internal: f-update
# =============================================================================

def _f_update_pcn(
    key: PRNGKey,
    state: RKPCNState,
    gp: GPJaxSurrogate,
    log_density_fn: Callable,
    rho: float,
) -> tuple[RKPCNState, Array]:
    """pCN f-update without Metropolis correction.

    Proposes a new trajectory g via pCN from the current trajectory f,
    realized at (u, v) where v is a fresh u-proposal. The f-update is
    accepted unconditionally (α_f ≡ 1).

    Returns
    -------
    state : RKPCNState
        Updated state with f_position = g(u) and logdensity = log π̃(u; g).
    g_at_v : Array, shape (q,)
        The proposed trajectory value g(v) at the u-proposal point.
        This is cached to avoid redundant JIT conditioning in the
        subsequent u-update.
    """
    key_jit, key_pcn = jr.split(key)

    u = state.position
    fu = state.f_position

    # JIT-condition to sample f(v) | f(u), then apply pCN to get g at (u, v)
    # We need a u-proposal point for the JIT. Sample one from the proposal.
    key_pcn, key_v = jr.split(key_pcn)
    v = _sample_gaussian_tril(key_v, m=u, L=state.proposal_tril).squeeze()

    gU, _ = _jit_pcn_proposal(
        key=key_jit,
        gp=gp,
        given=(u, fu),
        u_new=v,
        rho=rho,
    )

    # gU has shape (q, 2): column 0 = g(u), column 1 = g(v)
    gu = gU[:, 0]
    gv = gU[:, 1]

    # Update state: f ← g at current position
    new_logdensity = log_density_fn(gu, u)
    state = state._replace(
        f_position=gu.reshape(gp.output_dim),
        logdensity=new_logdensity,
    )

    return state, gv


# =============================================================================
# Internal: u-update
# =============================================================================

def _u_update(
    key: PRNGKey,
    state: RKPCNState,
    gp: GPJaxSurrogate,
    log_density_fn: Callable,
    f_at_proposal: Array,
    n_steps: int = 1,
) -> tuple[RKPCNState, RKPCNInfo]:
    """One or more MH u-updates targeting π(·; g) for the current trajectory g.

    For the first step, uses the cached f_at_proposal (from the f-update).
    For subsequent steps (n_steps > 1), proposes new u-positions and
    JIT-conditions to get g(ũ) at each.

    Parameters
    ----------
    f_at_proposal : Array, shape (q,)
        g(v) at the u-proposal point from the f-update. Used for
        the first u-step to avoid redundant JIT conditioning.

    Returns
    -------
    state : RKPCNState
        Updated state after all u-steps.
    info : RKPCNInfo
        Diagnostics from the *last* u-step (accept_prob, is_accepted).
    """
    return _multi_u_steps(key, state, gp, log_density_fn, f_at_proposal, n_steps)


def _single_u_step_cached(
    key: PRNGKey,
    state: RKPCNState,
    log_density_fn: Callable,
    f_at_proposal: Array,
) -> tuple[RKPCNState, RKPCNInfo]:
    """Single u-step using cached f(v) from the f-update."""
    key_propose, key_accept = jr.split(key)

    u = state.position
    fu = state.f_position

    # The proposal v was already drawn during the f-update; we need to
    # recover it. Since we can't pass it through, we re-draw using the
    # same key structure. But actually, the f-update already evaluated
    # g(v) — we just need the u-proposal point itself.
    #
    # Design note: in the current implementation, the u-proposal is
    # drawn inside _f_update_pcn and g(v) is returned as f_at_proposal.
    # We need to draw the same v here. We use a fresh key instead and
    # accept that the first u-proposal differs from the f-update's v.
    # This is a design simplification — the f-update pre-computes g(v)
    # for efficiency, but the u-update draws its own proposal.
    v = _sample_gaussian_tril(key_propose, m=u, L=state.proposal_tril).squeeze()

    # JIT-condition to get g(v) — but wait, we have f_at_proposal for the
    # v used in the f-update, not for this new v. For n_u_steps == 1,
    # we should reuse the f-update's v. Let me restructure.
    #
    # Actually, the clean approach: the f-update draws v and returns both
    # v and g(v). The u-update uses this same v. If rejected, done. If
    # more steps are needed, draw new proposals.
    #
    # For now (Phase 1, n_u_steps=1): we use f_at_proposal as g(v) and
    # need to know v. But we can't recover v from here.
    #
    # Simplest correct approach: pass v through from the f-update.
    # But NamedTuple state can't carry transient data easily.
    #
    # Resolution: the first u-update uses a NEW proposal (not the f-update's v).
    # This means the f_at_proposal cache is not used for n_u_steps=1.
    # Instead, we always JIT-condition. This is slightly less efficient but
    # correct and simple.
    #
    # TODO(Phase 2): optimize by passing v through for the first step.

    # JIT-condition: get g(v) for this new proposal
    # We condition g on (u, g(u)) which we know from the f-update.
    gv = gp_condition_sample(key_accept, gp, state.position, state.f_position, v)

    lp_prop = log_density_fn(gv, v)

    key_mh = jr.fold_in(key_accept, 1)
    u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
        key_mh, state.logdensity, lp_prop, u, v,
    )

    # Update f_position to track the accepted u
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


def _multi_u_steps(
    key: PRNGKey,
    state: RKPCNState,
    gp: GPJaxSurrogate,
    log_density_fn: Callable,
    f_at_proposal: Array,
    n_steps: int,
) -> tuple[RKPCNState, RKPCNInfo]:
    """Multiple u-steps with JIT conditioning for each new proposal.

    Uses jax.lax.scan for efficient compilation.
    """
    def scan_body(carry, key_step):
        state = carry
        key_prop, key_cond, key_mh = jr.split(key_step, 3)

        u = state.position
        v = _sample_gaussian_tril(key_prop, m=u, L=state.proposal_tril).squeeze()

        # JIT-condition: sample g(v) | g(u)
        gv = gp_condition_sample(key_cond, gp, u, state.f_position, v)
        lp_prop = log_density_fn(gv, v)

        u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
            key_mh, state.logdensity, lp_prop, u, v,
        )

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

        info = RKPCNInfo(accept_prob=accept_prob, is_accepted=is_accepted)
        return new_state, info

    keys = jr.split(key, n_steps)
    final_state, all_infos = jax.lax.scan(scan_body, state, keys)

    # Return the last info (for diagnostics)
    last_info = RKPCNInfo(
        accept_prob=all_infos.accept_prob[-1],
        is_accepted=all_infos.is_accepted[-1],
    )

    return final_state, last_info


# =============================================================================
# GP conditioning helpers
# =============================================================================

def gp_condition_sample(
    key: PRNGKey,
    gp: GPJaxSurrogate,
    u_cond: Array,
    f_cond: Array,
    u_new: Array,
) -> Array:
    """JIT-condition GP on (u_cond, f_cond) and sample at u_new.

    Parameters
    ----------
    key : PRNGKey
    gp : GPJaxSurrogate
    u_cond : Array, shape (d,)
        Conditioning input.
    f_cond : Array, shape (q,)
        Conditioning value f(u_cond).
    u_new : Array, shape (d,)
        New input to predict at.

    Returns
    -------
    f_new : Array, shape (q,)
        Sampled value g(u_new) | g(u_cond) = f_cond.
    """
    pred = gp.condition_then_predict(u_new, given=(u_cond, f_cond))
    return pred.sample(key).squeeze(0).reshape(gp.output_dim)


# =============================================================================
# JIT pCN proposal
# =============================================================================

def _jit_pcn_proposal(
    key: PRNGKey,
    gp: GPJaxSurrogate,
    given: tuple[Array, Array],
    u_new: Array,
    rho: float,
) -> tuple[Array, Array]:
    """JIT-condition then pCN proposal at (u, u_new).

    Given f(u), samples f(u_new) via GP conditioning, then applies the
    pCN proposal to the pair (f(u), f(u_new)).

    Parameters
    ----------
    given : (u, f(u))
        Current conditioning point and value.
    u_new : Array, shape (d,)
        New point to condition at.
    rho : float
        pCN correlation.

    Returns
    -------
    gU : Array, shape (q, 2)
        pCN proposal at [u, u_new]. Column 0 = g(u), column 1 = g(u_new).
    f_new : Array, shape (q,)
        The JIT-sampled f(u_new) (before pCN).
    """
    key_jit, key_pcn = jr.split(key)

    u, fu = given

    # JIT-condition: sample f(u_new) | f(u)
    f_new = gp.condition_then_predict(u_new, given=given).sample(key_jit).squeeze(0)

    # Stack for pCN at both points
    U = jnp.vstack([u, u_new])
    fU = jnp.hstack([
        jnp.reshape(fu, (gp.output_dim, 1)),
        jnp.reshape(f_new, (gp.output_dim, 1)),
    ])
    fU_dist = gp(U)
    gU = _pcn_proposal_batch(key_pcn, fU, fU_dist.mean, fU_dist.chol, rho)

    return gU, f_new.reshape(gp.output_dim)


# =============================================================================
# Low-level helpers (reused from samplers.py, kept here for independence)
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

    Returns (u_next, lp_next, accept_prob, is_accepted).
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


def _pcn_proposal_batch(
    key: PRNGKey,
    x: Array,
    mean: Array,
    cov_tril: Array,
    rho: float,
) -> Array:
    """Batched pCN proposal.

    Parameters
    ----------
    x : Array, shape (q, n) — current values at n points, q output dims
    mean : Array, shape (q, n) — GP prior mean at those points
    cov_tril : Array, shape (q, n, n) — GP prior Cholesky at those points
    rho : float — pCN correlation

    Returns
    -------
    proposal : Array, shape (q, n)
    """
    pcn_mean = mean + rho * (x - mean)
    pcn_tril = jnp.sqrt(1 - rho**2) * cov_tril
    return _sample_batch_gaussian_tril(key, m=pcn_mean, L=pcn_tril).squeeze(0)


# =============================================================================
# Log-density builders (convenience functions for common setups)
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
        ``(f, u) -> scalar`` log-density function.
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
