# uncprop/utils/distribution.py

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
        return self._numpyro_mvn.variance