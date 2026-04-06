"""
Tests for multi-chain MCMC utilities (uncprop/core/chain_combiner.py).

Tests cover:
1. Chain weighting (equal, mean_logdens, pritchard)
2. Failed chain detection
3. Duplicate mode identification
4. Chain combination
5. Initial position selection
"""
from jax import config
config.update('jax_enable_x64', True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from uncprop.core.chain_combiner import (
    compute_chain_weights,
    detect_failed_chains,
    identify_duplicate_modes,
    combine_chains,
    select_initial_positions,
    _farthest_point_sampling,
)


# ---------------------------------------------------------------------------
# Helper: mock chain results
# ---------------------------------------------------------------------------

def _make_chain_result(positions, logdensities, accept_rate=0.25):
    """Create a mock chain result dict."""
    ess = np.array([float(positions.shape[0]) / 2] * positions.shape[1])
    return {
        'positions': positions,
        'logdensities': logdensities,
        'accept_probs': np.full(len(logdensities), accept_rate),
        'post_burnin': positions,  # no burnin in test
        'ess': ess,
        'accept_rate': accept_rate,
        'init_position': positions[0],
        'runtime': 0.1,
        'chain_idx': 0,
    }


# ---------------------------------------------------------------------------
# Test: chain weighting
# ---------------------------------------------------------------------------

def test_equal_weights():
    """Equal weights should be 1/M and sum to 1."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
    ]
    w = compute_chain_weights(chains, method='equal')
    assert w.shape == (3,)
    assert np.allclose(w, 1/3)
    assert abs(w.sum() - 1.0) < 1e-10


def test_mean_logdens_weights():
    """Chain with higher mean log-density should get higher weight."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.full(100, -10.0)),
        _make_chain_result(np.random.randn(100, 2), np.full(100, 0.0)),
        _make_chain_result(np.random.randn(100, 2), np.full(100, 10.0)),
    ]
    w = compute_chain_weights(chains, method='mean_logdens')
    assert w[2] > w[1] > w[0]
    assert abs(w.sum() - 1.0) < 1e-10


def test_pritchard_weights():
    """Pritchard weights account for both mean and variance."""
    # Chain 1: low mean, low variance
    # Chain 2: low mean, high variance (should get boosted by variance term)
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.full(100, 0.0)),
        _make_chain_result(np.random.randn(100, 2),
                           np.concatenate([np.full(50, -5.0), np.full(50, 5.0)])),
    ]
    w = compute_chain_weights(chains, method='pritchard')
    assert w.shape == (2,)
    assert abs(w.sum() - 1.0) < 1e-10
    # Chain 2 has same mean (0) but higher variance → higher weight
    assert w[1] > w[0]


def test_weights_with_failed_mask():
    """Failed chains should get weight 0."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.full(100, 10.0)),
        _make_chain_result(np.random.randn(100, 2), np.full(100, 10.0)),
    ]
    w = compute_chain_weights(chains, method='equal',
                              failed_mask=np.array([True, False]))
    assert w[0] == 0.0
    assert abs(w[1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Test: failed chain detection
# ---------------------------------------------------------------------------

def test_detect_failed_low_ess():
    """Chain with very low ESS should be flagged."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
    ]
    # Manually set ESS to very low
    chains[0]['ess'] = np.array([5.0, 5.0])

    failed, diag = detect_failed_chains(chains, min_ess=10.0)
    assert failed[0] == True
    assert diag['n_failed'] == 1


def test_detect_failed_low_accept():
    """Chain with 0 acceptance should be flagged."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100),
                           accept_rate=0.0),
    ]
    failed, diag = detect_failed_chains(chains, min_accept=0.01)
    assert failed[0] == True


def test_detect_failed_passes_good_chain():
    """Good chain should not be flagged."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100),
                           accept_rate=0.25),
    ]
    chains[0]['ess'] = np.array([50.0, 50.0])

    failed, diag = detect_failed_chains(chains, min_ess=10.0, min_accept=0.01)
    assert failed[0] == False
    assert diag['n_failed'] == 0


# ---------------------------------------------------------------------------
# Test: duplicate mode identification
# ---------------------------------------------------------------------------

def test_identify_duplicate_modes_same():
    """Chains at the same location should be in the same mode."""
    center = np.array([0.5, 5.0])
    chains = [
        _make_chain_result(np.tile(center, (100, 1)) + 0.001 * np.random.randn(100, 2),
                           np.random.randn(100)),
        _make_chain_result(np.tile(center, (100, 1)) + 0.001 * np.random.randn(100, 2),
                           np.random.randn(100)),
    ]
    # Threshold in original coordinates — chains are ~0.001 apart
    labels = identify_duplicate_modes(chains, threshold=0.1)
    assert labels[0] == labels[1]


def test_identify_duplicate_modes_different():
    """Chains at different locations should be in different modes."""
    chains = [
        _make_chain_result(np.tile([0.0, 0.0], (100, 1)) + 0.01 * np.random.randn(100, 2),
                           np.random.randn(100)),
        _make_chain_result(np.tile([10.0, 10.0], (100, 1)) + 0.01 * np.random.randn(100, 2),
                           np.random.randn(100)),
    ]
    labels = identify_duplicate_modes(chains, threshold=0.1)
    assert labels[0] != labels[1]


# ---------------------------------------------------------------------------
# Test: chain combination
# ---------------------------------------------------------------------------

def test_combine_chains_count():
    """Total sample count should match sum of per-chain counts."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
        _make_chain_result(np.random.randn(200, 2), np.random.randn(200)),
    ]
    weights = np.array([0.5, 0.5])
    samples, sample_weights = combine_chains(chains, weights)

    assert samples.shape == (300, 2)
    assert sample_weights.shape == (300,)
    assert abs(sample_weights.sum() - 1.0) < 1e-10


def test_combine_chains_zero_weight():
    """Chain with weight 0 should contribute no samples."""
    chains = [
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
        _make_chain_result(np.random.randn(100, 2), np.random.randn(100)),
    ]
    weights = np.array([1.0, 0.0])
    samples, sample_weights = combine_chains(chains, weights)

    assert samples.shape == (100, 2)  # only chain 0


# ---------------------------------------------------------------------------
# Test: initial position selection
# ---------------------------------------------------------------------------

def test_select_positions_prior():
    """'prior' method should return correct shape within support."""
    from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep

    key = jr.key(42)
    posterior = generate_vsem_inv_prob_rep(
        key=key, par_names=('av', 'veg_init'),
        n_windows=12, noise_cov_tril=jnp.identity(12),
        n_days_per_window=30, observed_variable='lai')

    positions = select_initial_positions(
        jr.key(0), gp=None, prior=posterior.prior,
        n_chains=4, method='prior')

    assert positions.shape == (4, 2)


def test_farthest_point_sampling():
    """Selected points should be spread out."""
    candidates = np.random.randn(50, 2)
    selected = _farthest_point_sampling(candidates, 5)

    assert selected.shape == (5, 2)

    # Min pairwise distance should be positive
    from scipy.spatial.distance import pdist
    dists = pdist(selected)
    assert dists.min() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
