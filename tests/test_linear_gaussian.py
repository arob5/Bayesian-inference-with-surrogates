"""
Minimal viable test for the linear Gaussian experiment model code.

Tests the model layer (Gaussian distribution, inverse problem setup,
LinGaussInvProb) which lives in uncprop/models/linear_Gaussian/.
The experiment-level code (LinGaussTest, runner.py) requires the `modmcmc`
package, which is not tested here.
"""

import numpy as np

from uncprop.models.linear_Gaussian.Gaussian import Gaussian, kl_gauss, wasserstein_gauss
from uncprop.models.linear_Gaussian.LinGaussInvProb import LinGaussInvProb
from uncprop.models.linear_Gaussian.inverse_problem_setup import (
    make_inverse_problem,
    get_forward_model,
    gaussian_cov_mat,
)


def test_gaussian_basic():
    """Test Gaussian class: creation, sampling, affine transforms, KL/W2."""
    rng = np.random.default_rng(42)

    mean = np.array([1.0, 2.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    g = Gaussian(mean=mean, cov=cov, rng=rng)

    # Basic properties
    assert g.dim == 2
    assert np.allclose(g.mean, mean)
    assert np.allclose(g.cov, cov)

    # Sampling
    samp = g.sample(100)
    assert samp.shape == (100, 2)

    # Log density
    lp = g.log_p(mean)
    assert np.isfinite(lp)

    # KL divergence with itself should be ~0
    kl = g.kl(g)
    assert abs(kl) < 1e-10, f"KL with self should be ~0, got {kl}"

    # Wasserstein distance with itself should be ~0
    w2 = g.wasserstein(g)
    assert abs(w2) < 1e-10, f"W2 with self should be ~0, got {w2}"

    print("test_gaussian_basic: PASSED")


def test_gaussian_affine_inversion():
    """Test posterior computation via affine Gaussian inversion."""
    rng = np.random.default_rng(42)
    d = 3

    prior = Gaussian(mean=np.zeros(d), cov=np.eye(d), rng=rng)
    G = rng.normal(size=(2, d))
    noise_cov = 0.1 * np.eye(2)
    y = np.array([1.0, -0.5])

    post = prior.invert_affine_Gaussian(y, A=G, cov_noise=noise_cov)

    assert post.dim == d
    assert post.mean.shape == (d,)
    assert post.cov.shape == (d, d)

    # Posterior should be tighter than prior
    for j in range(d):
        assert post.cov[j, j] <= prior.cov[j, j] + 1e-10

    print("test_gaussian_affine_inversion: PASSED")


def test_lingauss_inv_prob():
    """Test LinGaussInvProb setup with synthetic data."""
    rng = np.random.default_rng(42)
    d = 5
    n = 3

    G = rng.normal(size=(n, d))
    m0 = np.zeros(d)
    C0 = np.eye(d)
    Sig = 0.1 * np.eye(n)

    inv_prob = LinGaussInvProb(rng, G, m0=m0, C0=C0, Sig=Sig)

    assert inv_prob.d == d
    assert inv_prob.n == n
    assert inv_prob.u_true is not None, "Ground truth not generated"
    assert inv_prob.y is not None, "Observations not generated"
    assert inv_prob.post is not None, "Posterior not computed"
    assert inv_prob.post.dim == d

    # Posterior covariance should be smaller than prior
    for j in range(d):
        assert inv_prob.post.cov[j, j] <= C0[j, j] + 1e-10

    print("test_lingauss_inv_prob: PASSED")


def test_make_inverse_problem():
    """Test the deconvolution inverse problem constructor."""
    rng = np.random.default_rng(42)

    inv_prob, g_conv, grid, idx_obs = make_inverse_problem(
        rng=rng, d=20, noise_sd=0.2,
        ker_length=5, ker_lengthscale=3, s=2,
    )

    assert inv_prob.d == 20
    assert inv_prob.G.shape[0] < inv_prob.G.shape[1]  # under-determined
    assert inv_prob.post is not None

    print("test_make_inverse_problem: PASSED")


if __name__ == '__main__':
    test_gaussian_basic()
    test_gaussian_affine_inversion()
    test_lingauss_inv_prob()
    test_make_inverse_problem()
    print("\nAll linear Gaussian tests passed!")
