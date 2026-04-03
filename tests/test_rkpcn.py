"""
Tests for the RKPCN v2 kernel (uncprop/core/rkpcn.py).

Tests cover:
1. Low-level helpers (MH accept/reject, pCN proposal)
2. Kernel construction and initialization
3. Single-step and multi-step execution
4. mcmc_loop compatibility
5. Multi-u-steps with iterative GP conditioning
"""
from jax import config
config.update('jax_enable_x64', True)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from uncprop.core.rkpcn import (
    RKPCNConfig,
    RKPCNState,
    RKPCNInfo,
    build_rkpcn_kernel,
    _mh_accept_reject,
    _pcn_proposal_univariate,
    _gp_sample_at,
)
from uncprop.core.samplers import mcmc_loop


# ---------------------------------------------------------------------------
# Fixture: VSEM GP surrogate
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_gp():
    """Create a GP surrogate from a small VSEM problem."""
    from uncprop.models.vsem.surrogate import fit_vsem_surrogate
    from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep

    key = jr.key(42)
    key_inv, key_surr = jr.split(key)

    posterior = generate_vsem_inv_prob_rep(
        key=key_inv,
        par_names=('av', 'veg_init'),
        n_windows=12,
        noise_cov_tril=jnp.identity(12),
        n_days_per_window=30,
        observed_variable='lai',
    )

    surrogate_post, _ = fit_vsem_surrogate(
        key=key_surr,
        posterior=posterior,
        n_design=4,
        surrogate_tag='clip_gp',
        design_method='lhc',
        jitter=1e-4,
    )

    return posterior, surrogate_post


# ---------------------------------------------------------------------------
# MH accept/reject
# ---------------------------------------------------------------------------

def test_mh_accept_reject_always_accepts():
    """When lp_prop >> lp_curr, should always accept."""
    key = jr.key(0)
    u_curr = jnp.array([1.0, 2.0])
    u_prop = jnp.array([3.0, 4.0])

    u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
        key, lp_curr=-100.0, lp_prop=0.0, u_curr=u_curr, u_prop=u_prop)

    assert is_accepted
    assert float(accept_prob) == 1.0
    assert jnp.allclose(u_next, u_prop)


def test_mh_accept_reject_always_rejects():
    """When lp_prop << lp_curr, should almost never accept."""
    key = jr.key(0)
    u_curr = jnp.array([1.0, 2.0])
    u_prop = jnp.array([3.0, 4.0])

    u_next, lp_next, accept_prob, is_accepted = _mh_accept_reject(
        key, lp_curr=0.0, lp_prop=-1000.0, u_curr=u_curr, u_prop=u_prop)

    assert not is_accepted
    assert float(accept_prob) < 1e-10
    assert jnp.allclose(u_next, u_curr)


def test_mh_accept_reject_nan_handling():
    """NaN log-density should be treated as -inf (rejected)."""
    key = jr.key(0)
    u_curr = jnp.array([1.0])
    u_prop = jnp.array([2.0])

    u_next, _, _, is_accepted = _mh_accept_reject(
        key, lp_curr=0.0, lp_prop=jnp.nan, u_curr=u_curr, u_prop=u_prop)

    assert not is_accepted


# ---------------------------------------------------------------------------
# pCN proposal
# ---------------------------------------------------------------------------

def test_pcn_univariate_rho_zero():
    """With rho=0, pCN proposal should be independent of f_u."""
    key = jr.key(42)
    q = 1
    f_u = jnp.array([10.0])
    mean = jnp.array([0.0])
    cov_tril = jnp.array([[[1.0]]])  # (1, 1, 1)

    g_u = _pcn_proposal_univariate(key, f_u, mean, cov_tril, rho=0.0)

    assert g_u.shape == (q,)
    # With rho=0: pcn_mean = mean = 0, so proposal should be ~N(0, 1)
    assert jnp.abs(g_u[0]) < 5.0  # very unlikely to be near f_u=10


def test_pcn_univariate_rho_near_one():
    """With rho≈1, pCN proposal should be very close to f_u."""
    key = jr.key(42)
    f_u = jnp.array([5.0])
    mean = jnp.array([0.0])
    cov_tril = jnp.array([[[1.0]]])

    g_u = _pcn_proposal_univariate(key, f_u, mean, cov_tril, rho=0.999)

    assert jnp.abs(g_u[0] - 5.0) < 0.5  # should be close to f_u


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

def test_kernel_construction(simple_gp):
    """build_rkpcn_kernel returns callable init_fn and kernel_fn."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    assert callable(init_fn)
    assert callable(kernel_fn)


def test_init_fn_returns_valid_state(simple_gp):
    """init_fn produces a valid RKPCNState with correct shapes."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99)
    init_fn, _ = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    assert isinstance(state, RKPCNState)
    assert state.position.shape == (d,)
    assert state.f_position.shape == (gp.output_dim,)
    assert jnp.isfinite(state.logdensity)
    assert state.proposal_tril.shape == (d, d)


# ---------------------------------------------------------------------------
# Single kernel step
# ---------------------------------------------------------------------------

def test_kernel_single_step(simple_gp):
    """kernel_fn produces valid state and info after one step."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    new_state, info = kernel_fn(jr.key(2), state)

    assert isinstance(new_state, RKPCNState)
    assert isinstance(info, RKPCNInfo)
    assert new_state.position.shape == (d,)
    assert jnp.isfinite(new_state.logdensity) or new_state.logdensity == -jnp.inf
    assert 0.0 <= float(info.accept_prob) <= 1.0


# ---------------------------------------------------------------------------
# mcmc_loop compatibility
# ---------------------------------------------------------------------------

def test_kernel_mcmc_loop_compatible(simple_gp):
    """kernel_fn works with mcmc_loop for multiple steps."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    n_steps = 50
    states, infos = mcmc_loop(jr.key(3), kernel_fn, state, num_samples=n_steps)

    assert states.position.shape == (n_steps, d)
    assert states.logdensity.shape == (n_steps,)
    assert infos.accept_prob.shape == (n_steps,)
    assert infos.is_accepted.shape == (n_steps,)

    accept_rate = float(jnp.mean(infos.accept_prob))
    assert accept_rate > 0.0, "No proposals accepted"
    print(f"  accept_rate over {n_steps} steps: {accept_rate:.4f}")


# ---------------------------------------------------------------------------
# Multi-u-steps
# ---------------------------------------------------------------------------

def test_kernel_multi_u_steps(simple_gp):
    """n_u_steps > 1 runs correctly with iterative GP conditioning."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99, n_u_steps=3)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    # Run a few macro-iterations manually
    for i in range(5):
        state, info = kernel_fn(jr.split(jr.key(i + 10))[0], state)
        assert jnp.isfinite(state.logdensity) or state.logdensity == -jnp.inf
        assert state.position.shape == (d,)

    print(f"  multi-u final accept_prob: {float(info.accept_prob):.4f}")


def test_multi_u_steps_mcmc_loop(simple_gp):
    """Multi-u-step kernel works with mcmc_loop."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99, n_u_steps=3)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    n_steps = 20
    states, infos = mcmc_loop(jr.key(3), kernel_fn, state, num_samples=n_steps)

    assert states.position.shape == (n_steps, d)
    print(f"  multi-u mcmc_loop accept_rate: {float(jnp.mean(infos.accept_prob)):.4f}")


# ---------------------------------------------------------------------------
# f-update correctness
# ---------------------------------------------------------------------------

def test_f_update_changes_f_position(simple_gp):
    """After one kernel step, f_position should change (pCN perturbation)."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.5)  # rho=0.5 gives large perturbation
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    key = jr.key(0)
    d = posterior.dim
    prop_cov = 0.01 * jnp.eye(d)
    state = init_fn(key, posterior.prior.sample(jr.key(1)), prop_cov)

    f_before = state.f_position.copy()
    new_state, _ = kernel_fn(jr.key(2), state)
    f_after = new_state.f_position

    # With rho=0.5, the pCN step should noticeably change f_position
    assert not jnp.allclose(f_before, f_after, atol=1e-6), \
        "f_position did not change after kernel step"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
