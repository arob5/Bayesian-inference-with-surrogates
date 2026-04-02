"""
Tests for the RKPCN v2 kernel (uncprop/core/rkpcn.py).

Phase 1 tests:
1. Kernel returns correct state/info types
2. Kernel is compatible with mcmc_loop
3. State initialization produces valid state
4. Multiple u-steps (n_u_steps > 1) runs without error
5. Integration test with VSEM surrogate (if available)
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
    _pcn_proposal_batch,
    gp_condition_sample,
)
from uncprop.core.samplers import mcmc_loop


# ---------------------------------------------------------------------------
# Fixtures: build a simple GP surrogate for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_gp():
    """Create a simple 1D GP surrogate for testing.

    Uses a small set of design points so the GP is fast.
    """
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
# Test: MH accept/reject helper
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
# Test: pCN proposal
# ---------------------------------------------------------------------------

def test_pcn_proposal_rho_zero():
    """With rho=0, pCN proposal should be independent of x."""
    key = jr.key(42)
    q, n = 1, 2
    x = jnp.ones((q, n)) * 10.0
    mean = jnp.zeros((q, n))
    cov_tril = jnp.eye(n).reshape(1, n, n)

    prop = _pcn_proposal_batch(key, x, mean, cov_tril, rho=0.0)

    # With rho=0: pcn_mean = mean, pcn_tril = cov_tril
    # So proposal should be centered at mean, not at x
    assert prop.shape == (q, n)
    # Proposal should be ~N(0, I), not near x=10
    assert jnp.all(jnp.abs(prop) < 5.0)  # very unlikely to be near 10


def test_pcn_proposal_rho_near_one():
    """With rho≈1, pCN proposal should be very close to x."""
    key = jr.key(42)
    q, n = 1, 2
    x = jnp.ones((q, n)) * 5.0
    mean = jnp.zeros((q, n))
    cov_tril = jnp.eye(n).reshape(1, n, n)

    prop = _pcn_proposal_batch(key, x, mean, cov_tril, rho=0.999)

    # With rho≈1: pcn_mean ≈ x, pcn_tril ≈ 0
    assert jnp.allclose(prop, x, atol=0.5)


# ---------------------------------------------------------------------------
# Test: RKPCN kernel construction and compatibility
# ---------------------------------------------------------------------------

def test_kernel_construction(simple_gp):
    """Test that build_rkpcn_kernel returns callable init_fn and kernel_fn."""
    posterior, surrogate_post = simple_gp

    from uncprop.core.rkpcn import build_log_density_vsem
    log_density_fn = build_log_density_vsem(posterior, surrogate_post)
    gp = surrogate_post.surrogate

    config = RKPCNConfig(rho=0.99)
    init_fn, kernel_fn = build_rkpcn_kernel(config, log_density_fn, gp)

    assert callable(init_fn)
    assert callable(kernel_fn)


def test_init_fn_returns_valid_state(simple_gp):
    """Test that init_fn produces a valid RKPCNState."""
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


def test_kernel_single_step(simple_gp):
    """Test that kernel_fn produces valid state and info after one step."""
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


def test_kernel_mcmc_loop_compatible(simple_gp):
    """Test that kernel_fn works with mcmc_loop."""
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

    # At least some proposals should be accepted
    accept_rate = float(jnp.mean(infos.accept_prob))
    assert accept_rate > 0.0, "No proposals accepted — check kernel"
    print(f"  accept_rate over {n_steps} steps: {accept_rate:.4f}")


def test_kernel_multi_u_steps(simple_gp):
    """Test that n_u_steps > 1 runs without error."""
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
    print(f"  multi-u accept_rate: {float(jnp.mean(infos.accept_prob)):.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
