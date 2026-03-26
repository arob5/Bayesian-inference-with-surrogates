# uncprop/utils/wasserstein.py
"""
Wasserstein distance utilities using entropic regularization (Sinkhorn).

Uses the ott-jax library for JAX-compatible optimal transport computation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem


def wasserstein2_sinkhorn(
    x_ref: jnp.ndarray,
    x_approx: jnp.ndarray,
    epsilon: float = 0.05,
    **kwargs
):
    """
    Compute entropic-regularized W2 distance between two empirical distributions.

    Args:
        x_ref:     (N, d) reference samples
        x_approx:  (M, d) approximating samples
        epsilon:    Sinkhorn regularization strength
        **kwargs:   forwarded to Sinkhorn()

    Returns:
        Scalar W2 distance
    """
    geom = pointcloud.PointCloud(x_ref, x_approx, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(**kwargs)
    out = solver(prob)
    return jnp.sqrt(out.reg_ot_cost)


def compute_wasserstein_comparison(
    samples: dict,
    reference_key: str,
    subsample: int | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    epsilon: float | None = None,
    sinkhorn_kwargs: dict | None = None
):
    """
    Compute whitened W2 distance from each set of samples to a reference.

    Whitens all samples using the reference distribution's mean and covariance
    (Mahalanobis metric), then computes entropic-regularized W2 via Sinkhorn.

    Args:
        samples:         dict mapping names to (n_samples, d) arrays
        reference_key:   key in samples dict to use as reference distribution
        subsample:       if set, randomly subsample this many points from each set
        key:             PRNG key for subsampling
        epsilon:         Sinkhorn regularization. If None, auto-computed from reference geometry.
        sinkhorn_kwargs: additional kwargs for Sinkhorn solver
                         (e.g., threshold=1e-6, max_iterations=5000, lse_mode=True)

    Returns:
        (results, epsilon) where results is a dict mapping non-reference names
        to scalar W2 distances, and epsilon is the regularization used.
    """
    sinkhorn_kwargs = sinkhorn_kwargs or {}
    ref_samples = samples[reference_key]
    n, d = ref_samples.shape

    # whitening matrix (Mahalanobis): Cov[(X-mu) @ W] = I
    mu_ref = jnp.mean(ref_samples, axis=0)
    cov_ref = jnp.cov(ref_samples, rowvar=False) + 1e-8 * jnp.eye(d)
    L = jax.scipy.linalg.cholesky(cov_ref, lower=True)
    W = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(d), lower=False)

    def whiten(samples):
        centered = samples - mu_ref
        return jnp.dot(centered, W)

    # Transform all chains using the reference's geometry
    samples = {
        k: whiten(v) for k, v in samples.items()
    }

    # choose regularization level using reference geometry
    if epsilon is None:
        ref_geom = pointcloud.PointCloud(
            samples[reference_key],
            samples[reference_key],
            epsilon=None
        )
        fixed_epsilon = ref_geom.epsilon
    else:
        fixed_epsilon = epsilon

    # optional subsampling
    if subsample is not None:
        for k, v in samples.items():
            key_choice, key = jr.split(key)
            idx = jr.choice(key_choice, v.shape[0], (subsample,), replace=False)
            samples[k] = v[idx]

    results = {}

    for name, x in samples.items():
        if name == reference_key:
            continue

        w2 = wasserstein2_sinkhorn(samples[reference_key], x, epsilon=fixed_epsilon, **sinkhorn_kwargs)
        results[name] = w2

    return results, fixed_epsilon
