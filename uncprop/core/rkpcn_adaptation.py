# uncprop/core/rkpcn_adaptation.py
"""
Adaptive proposal covariance for the RKPCN kernel.

Wraps the base RKPCN kernel with Haario-style covariance adaptation
and Robbins-Monro scale tuning. During the adaptation window, the
u-proposal covariance is updated periodically based on batches of
recent samples. After the adaptation window ends, the proposal is
frozen for the remainder of the run.

The adaptation reuses the same machinery as the standard adaptive
random-walk MH in ``uncprop.core.samplers`` (``AdaptationState``,
``update_adaptation``, ``AdaptationSettings``).

Design
------
The adaptive state extends the base ``RKPCNState`` with adaptation
bookkeeping fields. The kernel function contains two code paths
(selected via ``jax.lax.cond`` at each iteration):

1. **Adaptation active** (``step < adapt_end``): Run the base RKPCN
   kernel, accumulate samples in a batch buffer, and trigger a
   covariance/scale update every ``adapt_interval`` steps.

2. **Adaptation frozen** (``step >= adapt_end``): Run the base RKPCN
   kernel with a fixed proposal covariance.

Both paths are JIT-compiled inside ``jax.lax.scan``, so the kernel
is fully compatible with ``mcmc_loop``.

Usage
-----
::

    from uncprop.core.rkpcn import RKPCNConfig, build_log_density_fn
    from uncprop.core.rkpcn_adaptation import AdaptiveRKPCNConfig, build_adaptive_rkpcn_kernel
    from uncprop.core.samplers import mcmc_loop

    config = RKPCNConfig(rho=0.99, n_u_steps=1)
    adapt_config = AdaptiveRKPCNConfig(adapt_end=10_000, adapt_interval=50)

    init_fn, kernel_fn = build_adaptive_rkpcn_kernel(
        config, adapt_config, log_density_fn, gp)

    state = init_fn(key, position, prop_cov)
    states, infos = mcmc_loop(key, kernel_fn, state, num_samples=55_000)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.core.surrogate import GPJaxSurrogate
from uncprop.core.rkpcn import (
    RKPCNConfig,
    RKPCNState,
    RKPCNInfo,
    build_rkpcn_kernel,
)
from uncprop.core.samplers import (
    AdaptationState,
    AdaptationSettings,
    init_adaptation_state,
    update_adaptation,
    _proposal_tril_from_adaptation,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class AdaptiveRKPCNConfig:
    """Configuration for the adaptive proposal mechanism.

    Attributes
    ----------
    adapt_end : int
        Stop adapting after this many iterations. The proposal is
        frozen for all subsequent iterations.
    adapt_interval : int
        Number of iterations between adaptation updates. Samples
        are accumulated in a batch buffer and used to update the
        covariance every ``adapt_interval`` steps.
    target_accept : float
        Target acceptance rate for the Robbins-Monro scale tuning.
    gamma_exponent : float
        Decay rate for the adaptation step size:
        ``gamma = 1 / (n_adapted + 3)^gamma_exponent``.
        Larger values → faster decay → less aggressive late adaptation.
    scale_numerator : float
        Numerator in the scale update:
        ``log_scale += scale_numerator * gamma * (accept - target)``.
    adapt_cov : bool
        Whether to adapt the covariance shape (True) or only the
        global scale (False).
    jitter : float
        Regularization added to the covariance diagonal for
        numerical stability.
    """
    adapt_end: int = 10_000
    adapt_interval: int = 50
    target_accept: float = 0.234
    gamma_exponent: float = 0.8
    scale_numerator: float = 10.0
    adapt_cov: bool = True
    jitter: float = 1e-6


# =============================================================================
# State
# =============================================================================

class AdaptiveRKPCNState(NamedTuple):
    """State of the adaptive RKPCN sampler.

    Extends the base ``RKPCNState`` fields with adaptation bookkeeping.

    Attributes
    ----------
    position : Array, shape (d,)
        Current parameter position u.
    f_position : Array, shape (q,)
        Current GP trajectory value g(u).
    logdensity : float
        Log-density at the current state.
    proposal_tril : Array, shape (d, d)
        Lower Cholesky factor of the current u-proposal covariance.
        Updated during adaptation, frozen after ``adapt_end``.
    adapt_state : AdaptationState
        Running covariance, log scale, and adaptation counter.
    sample_history : Array, shape (adapt_interval, d)
        Buffer of recent u-positions for batch covariance estimation.
    accept_prob_history : Array, shape (adapt_interval,)
        Buffer of recent acceptance probabilities.
    step_in_batch : int
        Counter within the current batch (0 to adapt_interval-1).
    global_step : int
        Total iteration counter (for checking adapt_end).
    """
    position: Array
    f_position: Array
    logdensity: Array
    proposal_tril: Array
    adapt_state: AdaptationState
    sample_history: Array
    accept_prob_history: Array
    step_in_batch: int
    global_step: int


# =============================================================================
# Kernel builder
# =============================================================================

def build_adaptive_rkpcn_kernel(
    config: RKPCNConfig,
    adapt_config: AdaptiveRKPCNConfig,
    log_density_fn,
    gp: GPJaxSurrogate,
):
    """Build an adaptive RKPCN kernel compatible with ``mcmc_loop``.

    The returned kernel wraps the base RKPCN kernel and adds proposal
    adaptation during the first ``adapt_config.adapt_end`` iterations.

    Parameters
    ----------
    config : RKPCNConfig
        Base RKPCN configuration (rho, n_u_steps).
    adapt_config : AdaptiveRKPCNConfig
        Adaptation hyperparameters.
    log_density_fn : callable
        ``(f, u) -> scalar`` log-density function.
    gp : GPJaxSurrogate
        The GP surrogate model.

    Returns
    -------
    init_fn : callable
        ``(key, position, prop_cov) -> AdaptiveRKPCNState``
    kernel_fn : callable
        ``(key, state) -> (AdaptiveRKPCNState, RKPCNInfo)``
    """
    # Build the base kernel (non-adaptive)
    base_init_fn, base_kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    # Adaptation settings (reuses samplers.py machinery)
    adapt_settings = AdaptationSettings(
        adapt_interval=adapt_config.adapt_interval,
        target_accept=adapt_config.target_accept,
        adapt_cov=adapt_config.adapt_cov,
        adapt_scale=True,
        gamma_exponent=adapt_config.gamma_exponent,
        scale_numerator=adapt_config.scale_numerator,
        jitter=adapt_config.jitter,
    )

    adapt_end = adapt_config.adapt_end
    adapt_interval = adapt_config.adapt_interval

    # ------------------------------------------------------------------
    # Kernel function
    # ------------------------------------------------------------------
    def kernel_fn(key: PRNGKey, state: AdaptiveRKPCNState):
        # Extract base state for the base kernel
        base_state = RKPCNState(
            position=state.position,
            f_position=state.f_position,
            logdensity=state.logdensity,
            proposal_tril=state.proposal_tril,
        )

        # Run one step of the base RKPCN kernel
        new_base_state, info = base_kernel_fn(key, base_state)

        # --- Adaptation bookkeeping ---
        # Accumulate in batch buffers
        new_sample_history = jax.lax.dynamic_update_slice(
            state.sample_history,
            new_base_state.position.reshape(1, -1),
            (state.step_in_batch, 0)
        )
        new_accept_history = jax.lax.dynamic_update_slice(
            state.accept_prob_history,
            info.accept_prob.reshape(-1),
            (state.step_in_batch,)
        )

        next_step_in_batch = state.step_in_batch + 1
        next_global_step = state.global_step + 1

        # Check if we should trigger an adaptation update
        is_update_step = (
            (next_step_in_batch == adapt_interval)
            & (next_global_step <= adapt_end)
        )

        def _do_update(carry):
            adapt_st, acc_hist, samp_hist = carry
            avg_acc = jnp.mean(acc_hist)
            new_adapt = update_adaptation(
                adapt_st, samp_hist, avg_acc, adapt_settings
            )
            new_L = _proposal_tril_from_adaptation(new_adapt)
            return new_adapt, new_L

        def _no_update(carry):
            adapt_st, _, _ = carry
            return adapt_st, new_base_state.proposal_tril

        new_adapt_state, new_proposal_tril = jax.lax.cond(
            is_update_step,
            _do_update,
            _no_update,
            (state.adapt_state, new_accept_history, new_sample_history),
        )

        # Reset batch counter if adaptation triggered
        next_step_in_batch = jnp.where(is_update_step, 0, next_step_in_batch)

        # Build output state
        new_state = AdaptiveRKPCNState(
            position=new_base_state.position,
            f_position=new_base_state.f_position,
            logdensity=new_base_state.logdensity,
            proposal_tril=new_proposal_tril,
            adapt_state=new_adapt_state,
            sample_history=new_sample_history,
            accept_prob_history=new_accept_history,
            step_in_batch=next_step_in_batch,
            global_step=next_global_step,
        )

        return new_state, info

    # ------------------------------------------------------------------
    # Init function
    # ------------------------------------------------------------------
    def init_fn(key: PRNGKey, position: Array, prop_cov: Array):
        """Initialize adaptive RKPCN state.

        Parameters
        ----------
        key : PRNGKey
        position : Array, shape (d,)
            Initial parameter position.
        prop_cov : Array, shape (d, d)
            Initial u-proposal covariance. This is the starting point
            for adaptation.
        """
        # Initialize base state (gets f_position, logdensity, proposal_tril)
        base_state = base_init_fn(key, position, prop_cov)
        d = base_state.position.shape[0]

        # Initialize adaptation state
        adapt_state = init_adaptation_state(
            dim=d, initial_cov=prop_cov, n_chains=1
        )

        # Squeeze chain dimension (we run single-chain here)
        adapt_state = AdaptationState(
            cov_prop=adapt_state.cov_prop.squeeze(0),
            log_scale=adapt_state.log_scale.squeeze(0),
            times_adapted=adapt_state.times_adapted.squeeze(0),
        )

        return AdaptiveRKPCNState(
            position=base_state.position,
            f_position=base_state.f_position,
            logdensity=base_state.logdensity,
            proposal_tril=base_state.proposal_tril,
            adapt_state=adapt_state,
            sample_history=jnp.zeros((adapt_interval, d)),
            accept_prob_history=jnp.zeros(adapt_interval),
            step_in_batch=jnp.array(0, dtype=jnp.int64),
            global_step=jnp.array(0, dtype=jnp.int64),
        )

    return init_fn, kernel_fn


# =============================================================================
# Utilities
# =============================================================================

def extract_base_state(state: AdaptiveRKPCNState) -> RKPCNState:
    """Extract the base RKPCNState from an AdaptiveRKPCNState."""
    return RKPCNState(
        position=state.position,
        f_position=state.f_position,
        logdensity=state.logdensity,
        proposal_tril=state.proposal_tril,
    )


def get_adapted_proposal_cov(state: AdaptiveRKPCNState) -> Array:
    """Extract the current adapted proposal covariance from the state.

    Returns the full covariance ``scale^2 * C``.
    """
    scale = jnp.exp(state.adapt_state.log_scale)
    return scale**2 * state.adapt_state.cov_prop
