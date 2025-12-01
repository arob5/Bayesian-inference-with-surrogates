from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from jax.typing import ArrayLike
import jax.numpy as jnp

Array = jnp.ndarray

class Distribution(ABC):
    """
    A lightweight distribution class that is sufficient for the experiments in
    this paper. Uses basic shape/batch semantics: 
        - the event shape of the distribution is always a flat array, so is defined 
          by a single number, the dimension d (number of scalar elements in one value)
        - batches of points are stored as 2d arrays with shape (n, d)
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[tuple, tuple]:
        """
        Should return tuple of the form lower, upper where lower and upper
        are each tuples of length `dim` giving the dimension-by-dimension
        lower and upper bounds, respectively. Should be `-jnp.inf` or 
        `jnp.inf` for unbounded dimensions.
        """
        pass

    def sample(self, n: int = 1) -> Array:
        """ Out: (n,d)"""
        raise NotImplementedError('sample() not implemented for distribution object.')

    def log_density(self, x: ArrayLike) -> Array:
        """ In: (n,d) or (d,), Out: (n,)"""
        raise NotImplementedError('log_density() not implemented for distribution object.')
    
    @property
    def mean(self) -> Array:
        """ Return mean of shape (d,) """
        raise NotImplementedError('mean not implemented for distribution object.')

    @property
    def cov(self) -> Array:
        """ Return covariance matrix of shape (d,d) """
        raise NotImplementedError('cov not implemented for distribution object.')

    @property
    def variance(self) -> Array:
        """ Return marginal variances of shape (d,) """
        raise NotImplementedError('variance not implemented for distribution object.')

    @property
    def stdev(self) -> Array:
        """ Return marginal standard deviations of shape (d,) """
        return jnp.sqrt(self.variance)


class Prior(Distribution):
    """
    Simply a Distribution that is required to have names associated with each
    dimension.
    """

    @property
    @abstractmethod
    def par_names(self) -> list:
        """ Returns list of parameter names of length `self.dim` """
        pass

class Likelihood(ABC):
    @abstractmethod
    def log_density(self, x: ArrayLike) -> Array:
        """ In: (n,d), Out: (n,)"""
        pass

    def __call__(self, x: ArrayLike) -> Array:
        return self.log_density(x)


class Posterior(Distribution):
    """
    A Distribution representing the posterior distribution of a Bayesian inference
    problem, defined by specifying a Prior and Likelihood.
    """

    def log_density(self, x: ArrayLike):
        return self.prior.log_density(x) + self.likelihood(x)

    @property
    @abstractmethod
    def prior(self) -> Prior:
        pass

    @property
    @abstractmethod
    def likelihood(self) -> Likelihood:
        pass

    @prior.setter
    @abstractmethod
    def prior(self, value):
        pass

    @prior.setter
    @abstractmethod
    def prior(self, value):
        pass