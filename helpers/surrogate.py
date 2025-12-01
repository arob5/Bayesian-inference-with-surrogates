from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias
from jax.typing import ArrayLike
import jax.numpy as jnp

from inverse_problem import Distribution

Array: TypeAlias = jnp.ndarray

# predictive distribution of a surrogate at set of inputs
PredDist: TypeAlias = Distribution


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

    def mean(self, input: ArrayLike) -> Array:
        return self(input).mean
    
    def cov(self, input: ArrayLike) -> Array:
        return self(input).cov

    def variance(self, input: ArrayLike) -> Array:
        return self(input).variance

    def stdev(self, input: ArrayLike) -> Array:
        return self(input).stdev
    

class RandomDistribution(ABC):

    def log_density(self, input: ArrayLike) -> Distribution:
        raise NotImplementedError
    
    def deterministic_approx(self, method: str) -> Distribution:
        raise NotImplementedError
    
    def expected_posterior(self) -> Distribution:
        raise NotImplementedError

    def expected_unnormalized_posterior(self) -> Distribution:
        raise NotImplementedError

    def expected_


class SurrogateDistribution(ABC):
    """
    This class represents a random probability distribution, where the
    randomness stems from a Surrogate. A typical example is a
    surrogate-based approximation to an underlying forward model or 
    the log-density, which induces a random distribution.
    """

    @property
    @abstractmethod
    def surrogate(self) -> Surrogate:
        pass

    @surrogate.setter
    @abstractmethod
    def surrogate(self, value: Surrogate):
        pass

    def log_density(self, input: ArrayLike | PredDist) -> PredDist:
        """
        Returns Distribution representing the surrogate-based prediction
        of the log-density at the input points `input`. If `input` is 
        itself a Distribution, then it should be interpreted as the 
        surrogate predictive distribution at those points, which prevents
        re-computing the predictions if they have already been computed.
        """
        raise NotImplementedError
        
    def expected_surrogate_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_log_density_approx(self) -> Distribution:
        raise NotImplementedError

    def expected_normalized_density_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_density_approx(self) -> Distribution:
        raise NotImplementedError


class RandomDistributionExpectation(Distribution):
    pass
