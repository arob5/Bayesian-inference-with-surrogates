"""
Tests for GPJaxSurrogate conditioning and prediction.

Verifies:
1. condition() is consistent with condition_then_predict()
2. Iterative one-at-a-time conditioning matches batch conditioning
3. Conditioned GP supports further conditioning (chaining)
4. Conditioned GP predictions have correct shapes
5. Conditioning reduces predictive variance
"""
from jax import config
config.update('jax_enable_x64', True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from uncprop.core.surrogate import GPJaxSurrogate


# ---------------------------------------------------------------------------
# Fixture: build a GP surrogate from the VSEM model
# ---------------------------------------------------------------------------

@pytest.fixture
def vsem_gp():
    """Create a GPJaxSurrogate from a small VSEM problem."""
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

    return surrogate_post.surrogate, posterior


# ---------------------------------------------------------------------------
# Test: condition() consistency with condition_then_predict()
# ---------------------------------------------------------------------------

def test_condition_matches_condition_then_predict(vsem_gp):
    """condition().predict(x) should match condition_then_predict(x, given)."""
    gp, posterior = vsem_gp
    key = jr.key(100)

    # Conditioning data: one new point
    x_cond = jnp.array([[0.5, 5.0]])
    y_cond = gp(x_cond).sample(key).squeeze(0)  # (1, q)

    # Test points
    x_test = jnp.array([[0.3, 3.0], [0.7, 7.0], [0.9, 9.0]])

    # Approach 1: condition_then_predict
    pred_ctp = gp.condition_then_predict(x_test, given=(x_cond, y_cond))

    # Approach 2: condition() then predict()
    gp_cond = gp.condition(given=(x_cond, y_cond))
    pred_cond = gp_cond.predict(x_test)

    # Means should match
    assert jnp.allclose(pred_ctp.mean, pred_cond.mean, atol=1e-10), \
        f"Mean mismatch: max diff = {jnp.max(jnp.abs(pred_ctp.mean - pred_cond.mean))}"

    # Covariances should match
    assert jnp.allclose(pred_ctp.cov, pred_cond.cov, atol=1e-10), \
        f"Cov mismatch: max diff = {jnp.max(jnp.abs(pred_ctp.cov - pred_cond.cov))}"

    print("  condition() matches condition_then_predict(): PASSED")


def test_condition_matches_ctp_multiple_points(vsem_gp):
    """Same test but conditioning on multiple points at once."""
    gp, posterior = vsem_gp
    key = jr.key(200)

    # Conditioning data: 3 new points
    x_cond = jnp.array([[0.2, 2.0], [0.5, 5.0], [0.8, 8.0]])
    key1, key2 = jr.split(key)
    y_cond = gp(x_cond).sample(key1).squeeze(0)  # (3, q)

    # Test points
    x_test = jnp.array([[0.4, 4.0], [0.6, 6.0]])

    pred_ctp = gp.condition_then_predict(x_test, given=(x_cond, y_cond))
    gp_cond = gp.condition(given=(x_cond, y_cond))
    pred_cond = gp_cond.predict(x_test)

    assert jnp.allclose(pred_ctp.mean, pred_cond.mean, atol=1e-10)
    assert jnp.allclose(pred_ctp.cov, pred_cond.cov, atol=1e-10)

    print("  condition() matches ctp (multiple points): PASSED")


# ---------------------------------------------------------------------------
# Test: iterative vs batch conditioning
# ---------------------------------------------------------------------------

def test_iterative_conditioning_matches_batch(vsem_gp):
    """Conditioning one point at a time should match conditioning all at once."""
    gp, posterior = vsem_gp
    key = jr.key(300)

    # Three conditioning points
    x1 = jnp.array([[0.2, 2.0]])
    x2 = jnp.array([[0.5, 5.0]])
    x3 = jnp.array([[0.8, 8.0]])

    key1, key2, key3, key_test = jr.split(key, 4)

    # Sample consistent y-values from the GP
    # First sample at all 3 points jointly, then split
    x_all = jnp.vstack([x1, x2, x3])
    y_all = gp(x_all).sample(key1).squeeze(0)  # (3, q)
    y1, y2, y3 = y_all[0:1], y_all[1:2], y_all[2:3]

    # Test points
    x_test = jnp.array([[0.4, 4.0], [0.6, 6.0]])

    # Approach 1: batch conditioning
    gp_batch = gp.condition(given=(x_all, y_all))
    pred_batch = gp_batch.predict(x_test)

    # Approach 2: iterative conditioning (one at a time)
    gp_iter1 = gp.condition(given=(x1, y1))
    gp_iter2 = gp_iter1.condition(given=(x2, y2))
    gp_iter3 = gp_iter2.condition(given=(x3, y3))
    pred_iter = gp_iter3.predict(x_test)

    # Should match (up to floating-point differences from order of operations)
    assert jnp.allclose(pred_batch.mean, pred_iter.mean, atol=1e-8), \
        f"Mean mismatch: max diff = {jnp.max(jnp.abs(pred_batch.mean - pred_iter.mean))}"
    assert jnp.allclose(pred_batch.cov, pred_iter.cov, atol=1e-8), \
        f"Cov mismatch: max diff = {jnp.max(jnp.abs(pred_batch.cov - pred_iter.cov))}"

    print("  Iterative conditioning matches batch: PASSED")


# ---------------------------------------------------------------------------
# Test: conditioning reduces variance
# ---------------------------------------------------------------------------

def test_conditioning_reduces_variance(vsem_gp):
    """Predictive variance should decrease (or stay same) after conditioning."""
    gp, posterior = vsem_gp
    key = jr.key(400)

    x_test = jnp.array([[0.5, 5.0]])

    # Unconditioned prediction
    pred_prior = gp.predict(x_test)
    var_prior = pred_prior.variance

    # Condition on a nearby point
    x_cond = jnp.array([[0.45, 4.5]])
    y_cond = gp(x_cond).sample(key).squeeze(0)

    gp_cond = gp.condition(given=(x_cond, y_cond))
    pred_cond = gp_cond.predict(x_test)
    var_cond = pred_cond.variance

    # Variance should decrease (conditioning adds information)
    assert jnp.all(var_cond <= var_prior + 1e-10), \
        f"Variance increased: prior={var_prior}, conditioned={var_cond}"

    print(f"  Variance reduced: {float(var_prior.mean()):.6f} -> {float(var_cond.mean()):.6f}")


# ---------------------------------------------------------------------------
# Test: conditioned GP has correct shapes
# ---------------------------------------------------------------------------

def test_conditioned_gp_shapes(vsem_gp):
    """Conditioned GP predictions should have consistent shapes."""
    gp, posterior = vsem_gp
    key = jr.key(500)

    q = gp.output_dim
    d = gp.input_dim

    # Condition on 2 points
    x_cond = jnp.array([[0.3, 3.0], [0.7, 7.0]])
    y_cond = gp(x_cond).sample(key).squeeze(0)  # (2, q)

    gp_cond = gp.condition(given=(x_cond, y_cond))

    # Check attributes
    assert gp_cond.output_dim == q
    assert gp_cond.input_dim == d
    assert gp_cond.design.n == gp.design.n + 2
    assert gp_cond.P.shape == (q, gp.design.n + 2, gp.design.n + 2)

    # Predict at single point
    x1 = jnp.array([[0.5, 5.0]])
    pred1 = gp_cond.predict(x1)
    assert pred1.mean.shape == (q, 1) or pred1.mean.shape == (1,), \
        f"Unexpected mean shape: {pred1.mean.shape}"

    # Predict at multiple points
    x3 = jnp.array([[0.2, 2.0], [0.5, 5.0], [0.8, 8.0]])
    pred3 = gp_cond.predict(x3)
    # Shape depends on q: if q=1, mean is (3,); if q>1, mean is (q, 3)
    print(f"  Shapes: q={q}, pred1.mean={pred1.mean.shape}, pred3.mean={pred3.mean.shape}")

    print("  Conditioned GP shapes: PASSED")


# ---------------------------------------------------------------------------
# Test: conditioned GP supports sampling
# ---------------------------------------------------------------------------

def test_conditioned_gp_sampling(vsem_gp):
    """Samples from conditioned GP should have correct shapes."""
    gp, posterior = vsem_gp
    key = jr.key(600)

    x_cond = jnp.array([[0.5, 5.0]])
    y_cond = gp(x_cond).sample(key).squeeze(0)

    gp_cond = gp.condition(given=(x_cond, y_cond))

    # Sample at a new point
    x_new = jnp.array([[0.6, 6.0]])
    key_samp = jr.key(601)
    pred = gp_cond.predict(x_new)
    samp = pred.sample(key_samp)

    assert samp.ndim >= 1
    assert jnp.all(jnp.isfinite(samp)), "Samples contain non-finite values"

    print(f"  Sample shape: {samp.shape}")
    print("  Conditioned GP sampling: PASSED")


# ---------------------------------------------------------------------------
# Test: conditioning at design point collapses variance
# ---------------------------------------------------------------------------

def test_conditioning_at_design_point(vsem_gp):
    """Conditioning at an existing design point should nearly collapse variance."""
    gp, posterior = vsem_gp
    key = jr.key(700)

    # Pick the first design point
    x_design = gp.design.X[0:1]  # (1, d)
    y_design = gp.design.y[0:1]  # (1, q)

    # Predict at a nearby test point before and after conditioning
    x_test = x_design + 0.001  # slightly offset

    pred_before = gp.predict(x_test)
    var_before = pred_before.variance

    # Condition on the design point (with its actual y-value)
    gp_cond = gp.condition(given=(x_design, y_design))
    pred_after = gp_cond.predict(x_test)
    var_after = pred_after.variance

    # Variance near a design point should be very small
    # (it's already small from the original design, but should stay small)
    assert jnp.all(var_after <= var_before + 1e-10)

    print(f"  Var near design point: before={float(var_before.mean()):.8f}, "
          f"after={float(var_after.mean()):.8f}")


# ---------------------------------------------------------------------------
# Test: chained conditioning (3 levels)
# ---------------------------------------------------------------------------

def test_triple_chained_conditioning(vsem_gp):
    """Three levels of chaining: gp → gp1 → gp2 → gp3."""
    gp, posterior = vsem_gp
    key = jr.key(800)

    keys = jr.split(key, 4)

    x1 = jnp.array([[0.2, 2.0]])
    y1 = gp(x1).sample(keys[0]).squeeze(0)
    gp1 = gp.condition(given=(x1, y1))

    x2 = jnp.array([[0.5, 5.0]])
    y2 = gp1(x2).sample(keys[1]).squeeze(0)
    gp2 = gp1.condition(given=(x2, y2))

    x3 = jnp.array([[0.8, 8.0]])
    y3 = gp2(x3).sample(keys[2]).squeeze(0)
    gp3 = gp2.condition(given=(x3, y3))

    # Verify chain sizes
    assert gp1.design.n == gp.design.n + 1
    assert gp2.design.n == gp.design.n + 2
    assert gp3.design.n == gp.design.n + 3

    # Predictions should be valid
    x_test = jnp.array([[0.4, 4.0]])
    pred = gp3.predict(x_test)
    assert jnp.all(jnp.isfinite(pred.mean))
    assert jnp.all(pred.variance > 0)

    print(f"  Triple chain: design sizes {gp.design.n} → {gp1.design.n} "
          f"→ {gp2.design.n} → {gp3.design.n}")
    print("  Triple chained conditioning: PASSED")


# ---------------------------------------------------------------------------
# Test: original GP is not modified by condition()
# ---------------------------------------------------------------------------

def test_condition_does_not_modify_original(vsem_gp):
    """condition() should not alter the original GP's state."""
    gp, posterior = vsem_gp
    key = jr.key(900)

    # Record original state
    P_orig = gp.P.copy()
    n_orig = gp.design.n

    x_test = jnp.array([[0.5, 5.0]])
    pred_orig = gp.predict(x_test)

    # Condition
    x_cond = jnp.array([[0.3, 3.0]])
    y_cond = gp(x_cond).sample(key).squeeze(0)
    gp_cond = gp.condition(given=(x_cond, y_cond))

    # Original should be unchanged
    assert gp.design.n == n_orig
    assert gp.P.shape == P_orig.shape
    assert jnp.allclose(gp.P, P_orig)

    pred_after = gp.predict(x_test)
    assert jnp.allclose(pred_orig.mean, pred_after.mean, atol=1e-12)

    print("  Original GP not modified: PASSED")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
