from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias
from collections.abc import Callable
import jax.numpy as jnp
from gpjax import Dataset
from numpyro.distributions import Distribution as NumpyroDistribution

from uncprop.core.inverse_problem import (
    Distribution,
    DistributionFromDensity,
    Prior,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike

# predictive distribution of a surrogate at set of inputs
PredDist: TypeAlias = Distribution | NumpyroDistribution


def construct_design(key: PRNGKey,
                     design_method: str, 
                     n_design: int, 
                     prior: Prior, 
                     f: Callable) -> Dataset:
    """ Construct design (training) data for training a surrogate

    Sample design inputs from prior, then evaluate target function f to
    construct design outputs. Return design as gpjax Dataset object.
    """

    if design_method == 'lhc':
        x_design = prior.sample_lhc(key, n_design)
    elif design_method == 'uniform':
        x_design = prior.sample(key, n_design)
    else:
        raise ValueError(f'Invalid design method {design_method}')

    x_design = jnp.asarray(x_design)
    y_design = jnp.asarray(f(x_design))

    if y_design.ndim < 2:
        y_design = y_design.reshape(-1, 1)

    return Dataset(X=x_design, y=y_design)


class Surrogate(ABC):
    """
    A Surrogate `f` is defined as a callable, such that `f(input)` returns
    a Distribution object representing the predictive distribution at the 
    set of points `input`. Note that the dimension of the returned distribution
    will depend on the number of input points.

    This class provides convenience functions to extract certain properties
    of the predictive distribution (mean, cov, variance, standard deviation),
    which are by default simple aliases; e.g., f.mean(input) simply calls
    f(input).mean. If there are more efficient ways to compute these quantities
    without realizing the full predictive distribution, then these methods should
    be overwritten.
    """

    @abstractmethod
    def __call__(self, input: ArrayLike) -> PredDist:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    def sample_trajectory(self) -> Callable[[ArrayLike], Array]:
        """ Returns a function representing a trajectory of the surrogate 
        
        The function should be vectorized so that it can be evaluated at (n,d)
        input batches, giving the values of the surrogate trajectory at the n points.  
        """
        raise NotImplementedError('sample_trajectory() not implemented by Surrogate object.')

    def mean(self, input: ArrayLike) -> Array:
        return self(input).mean
    
    def cov(self, input: ArrayLike) -> Array:
        return self(input).cov

    def variance(self, input: ArrayLike) -> Array:
        return self(input).variance

    def stdev(self, input: ArrayLike) -> Array:
        return self(input).stdev
    
    
class SurrogateDistribution(ABC):
    """
    This class represents a random probability distribution, where the
    randomness stems from a Surrogate. A typical example is a
    surrogate-based approximation to an underlying forward model or 
    the log-density, which induces a random distribution.

    More formally, this class represents the random distribution:
        pi(u; f) = p(u; f) / Z(f), Z(f) = int_u p(u; f) du
    where f is a (random) surrogate. Therefore, the p(u; f), Z(f),
    and pi(u; f) represent the random unnormalized density, random
    normalizing constant, and random normalized density, respectively.
    """

    @property
    @abstractmethod
    def surrogate(self) -> Surrogate:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[tuple, tuple] | None:
        pass

    def log_density_from_pred(self, pred: PredDist) -> PredDist:
        """
        The log density parameterized as a function of the surrogate
        predictive distribution. `pred` represents surrogate predictions
        at a set of points. This function returns a Distribution representing
        the pushforward of the surrogate predictions through the log density
        map. 
        """
        raise NotImplementedError
    
    def log_density(self, input: ArrayLike) -> PredDist:
        """ Same return type as `log_density_from_pred` but first computes
            the surrogate predictions at a set of inputs. Then pushes the 
            distribution through the log density map.
        """
        pred = self.surrogate(input)
        return self.log_density_from_pred(pred)
    
    def sample_trajectory(self) -> Distribution:
        """ Returns a Distribution representing a trajectory of the random distribution 
        
        Since a SurrogateDistribution is a random distribution, a trajectory of a 
        SurrogateDistribution is a deterministic Distribution. Subclasses will often 
        implement this by sampling a log density trajectory and then returning a 
        DistributionFromDensity. However, there may be other cases.
        """
        raise NotImplementedError('sample_trajectory() not implemented by SurrogateDistribution object.')

    def expected_surrogate_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_log_density_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_density_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_normalized_density_approx(self, method, **method_kwargs) -> Distribution:
        raise NotImplementedError
    
    def _expected_normalized_imputation(self, 
                                        n_trajectory: int = 100, 
                                        n_samp_per_trajectory: int = 1):
        raise NotImplementedError


class LogDensGPSurrogate(SurrogateDistribution):
    """
    A SurrogateDistribution defined by a Gaussian log-density surrogate.
    Concretely, the random distribution is:
        pi(u; f) = exp{f(u)} / Z(f), Z(f) = int_u exp{f(u)} du, where f(u) ~ N(m(u), C(u))

    That is, f is a Gaussian process.
    """

    def __init__(self, log_dens: Surrogate, support: tuple[tuple, tuple] | None = None):
        if not isinstance(log_dens, Surrogate):
            raise ValueError(f'LogDensGPSurrogate requires `log_dens` to be a Surrogate, got {type(log_dens)}')
        self._surrogate = log_dens
        self._support = support

    @property
    def surrogate(self):
        return self._surrogate
    
    @property
    def dim(self):
        return self.surrogate.input_dim
    
    @property
    def support(self):
        return self._support
    

    def log_density_from_pred(self, pred: PredDist):
        """ Surrogate predictions are log-density predictions """
        return pred
    
    def expected_surrogate_approx(self) -> DistributionFromDensity:
        """ Plug-in surrogate mean as log-density approximation. """
        surrogate_mean = lambda x: self.surrogate.mean(x)

        return DistributionFromDensity(log_dens=surrogate_mean,
                                       dim=self.surrogate.input_dim)
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        """ Surrogate is the log-density, so equivalent to expected_surrogate_approx. """
        return self.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        """ Density surrogate exp{f(u)} is log-normal, so expectation is log-normal mean """
        def log_expected_dens(x):
            pred = self.surrogate(x)
            return pred.mean + 0.5 * pred.variance
        
        return DistributionFromDensity(log_dens=log_expected_dens,
                                       dim=self.surrogate.input_dim)