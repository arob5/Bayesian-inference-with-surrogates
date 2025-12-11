# uncprop/utils/distribution.py

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.stats import norm

from numpyro.distributions import MultivariateNormal

from uncprop.custom_types import Array, ArrayLike, PRNGKey
from uncprop.core.inverse_problem import Distribution

# -----------------------------------------------------------------------------
# Wrap numpyro MultivariateNormal as Distribution
# -----------------------------------------------------------------------------

class GaussianFromNumpyro(Distribution):
    def __init__(self, dist: MultivariateNormal):
        assert isinstance(dist, MultivariateNormal)
        assert dist.batch_shape == ()

        self._numpyro_mvn = dist

    @property
    def dim(self) -> int:
        return self._numpyro_mvn.event_shape[0]

    @property
    def support(self) -> tuple[tuple, tuple] | None:
        return None

    def sample(self, key: PRNGKey, n: int = 1) -> Array:
        """ Out: (n,d)"""
        return self._numpyro_mvn.sample(key, sample_shape=(n,))

    def log_density(self, x: ArrayLike) -> Array:
        """ In: (n,d) or (d,), Out: (n,)"""
        return self._numpyro_mvn.log_prob(x)
    
    @property
    def mean(self) -> Array:
        """ Return mean of shape (d,) """
        return self._numpyro_mvn.mean

    @property
    def cov(self) -> Array:
        """ Return covariance matrix of shape (d,d) """
        return self._numpyro_mvn.covariance_matrix

    @property
    def variance(self) -> Array:
        """ Return marginal variances of shape (d,) """
        return jnp.diag(self.cov)
    

# -----------------------------------------------------------------------------
# Clipped Gaussian and LogNormal mean
# -----------------------------------------------------------------------------

def clipped_gaussian_mean(m: ArrayLike,
                          s2: ArrayLike,
                          b: ArrayLike) -> Array:
    """Compute E[min(X, b)] for X ~ Normal(m, s2), vectorized and JAX-compatible.

    The inputs `m`, `s2`, and `b` may be scalars or arrays of the same shape
    (or broadcastable shapes). The result is the elementwise expectation
    E[min(X, b)] computed using the analytic formula:

        E[min(X, b)] = b + (m - b) * Phi(z) - sigma * phi(z),
        where z = (b - m) / sigma, sigma = sqrt(s2).

    The degenerate case s2 == 0 is handled: E[min(X,b)] = min(m, b).

    Args:
        m: Mean(s) of the normal distribution (scalar or array).
        s2: Variance(s) of the normal distribution (scalar or array).
        b: Upper clipping bound(s) (scalar or array).

    Returns:
        An array (or scalar) of the same broadcasted shape containing
        E[min(X, b)].
    """
    m = jnp.asarray(m)
    s2 = jnp.asarray(s2)
    b = jnp.asarray(b)

    # safe_sigma used only to compute z when sigma == 0 (avoid division by zero).
    # Value chosen arbitrarily (1.0) because z won't be used in the degenerate branch.
    sigma = jnp.sqrt(s2)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    z = (b - m) / safe_sigma

    # Standard normal CDF via erf for numerical stability and JAX compatibility
    sqrt2 = jnp.sqrt(2.0)
    Phi = 0.5 * (1.0 + jsp.special.erf(z / sqrt2))

    # Standard normal PDF
    inv_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)
    phi = inv_sqrt_2pi * jnp.exp(-0.5 * z * z)

    # analytic expectation (valid when sigma > 0)
    result_continuous = b + (m - b) * Phi - sigma * phi

    # degenerate case: sigma == 0 -> X == m almost surely => E[min(X,b)] = min(m, b)
    result = jnp.where(sigma > 0.0, result_continuous, jnp.minimum(m, b))

    return result


def log_clipped_lognormal_mean(m: ArrayLike, s2: ArrayLike, b: ArrayLike) -> Array:
    """Compute log E[min(exp(x), exp(b))] for x ~ N(m, s2), JAX-compatible and vectorized.

    The inputs `m`, `s2`, and `b` may be scalars or arrays (broadcastable shapes).
    The returned value is the elementwise natural logarithm of the expectation:

        log E[min(e^x, e^b)].

    Analytic decomposition used:

        E[min(e^x, e^b)] = e^b * P(x >= b) + E[e^x 1_{x < b}]
                          = e^b * (1 - Φ(z1)) + exp(m + s2/2) * Φ(z2),

    where z1 = (b - m) / σ and z2 = (b - m - s2) / σ, with σ = sqrt(s2).
    We compute the logarithm using stable `log` and `logaddexp` operations and
    stable evaluations of log-Φ and log-(1-Φ) in the tails.

    Degenerate case (s2 == 0): x == m deterministically so
        E[min(e^x, e^b)] = exp(min(m, b))  =>  log E = min(m, b).

    Args:
        m: Mean(s) of the normal variable x (scalar or array).
        s2: Variance(s) of the normal variable x (scalar or array).
        b: Log clipping bound(s) (returns log E[min(e^x, e^b)]).

    Returns:
        An array (or scalar) with the same broadcasted shape containing
        log E[min(e^x, e^b)].
    """
    m = jnp.asarray(m)
    s2 = jnp.asarray(s2)
    b = jnp.asarray(b)

    # sigma, but guard zero variance to avoid division by zero
    sigma = jnp.sqrt(s2)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    # z-values
    z1 = (b - m) / safe_sigma
    z2 = (b - m - s2) / safe_sigma

    # log term corresponding to e^b * (1 - Phi(z1))
    log_term1 = b + norm.logsf(z1)

    # log term corresponding to exp(m + s2/2) * Phi(z2)
    log_term2 = m + 0.5 * s2 + norm.logcdf(z2)

    # log-sum-exp of two positive contributions (elementwise)
    log_sum = jnp.logaddexp(log_term1, log_term2)

    # For degenerate case (s2 == 0): log E = min(m, b)
    result = jnp.where(sigma > 0.0, log_sum, jnp.minimum(m, b))

    return result

    