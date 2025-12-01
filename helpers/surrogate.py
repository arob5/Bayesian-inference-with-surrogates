from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from jax.typing import ArrayLike
import jax.numpy as jnp

from inverse_problem import Distribution

Array = jnp.ndarray


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
    def __call__(self, input: ArrayLike) -> Distribution:
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
    pass