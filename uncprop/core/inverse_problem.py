from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, TypeAlias, Any
from collections.abc import Callable
import jax.numpy as jnp
import jax.random as jr

from uncprop.core.samplers import (
    init_nuts_kernel,
    mcmc_loop,
    stack_dict_arrays,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike


class Distribution(ABC):
    """
    A minimal distribution class that is sufficient for the experiments in
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
    def support(self) -> tuple[tuple, tuple] | None:
        """
        Should return tuple of the form lower, upper where lower and upper
        are each tuples of length `dim` giving the dimension-by-dimension
        lower and upper bounds, respectively. Should be `-jnp.inf` or 
        `jnp.inf` for unbounded dimensions. None is interpreted as unconstrained.
        """
        pass

    def sample(self, key: PRNGKey, n: int = 1) -> Array:
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
    

@runtime_checkable
class LogDensity(Protocol):
    def __call__(self, x: ArrayLike) -> Array:
        """ In: (n,d), Out: (n,)"""
        pass


class DistributionFromDensity(Distribution):
    """
    Convenience class for constructing a Distribution given its (unnormalized)
    log density.
    """
    def __init__(self, 
                 log_dens: LogDensity, 
                 dim: int, 
                 support: tuple[tuple, tuple] | None = None):
        
        if not isinstance(log_dens, LogDensity):
            raise ValueError(f'DistributionFromDensity expects LogDensity, got {type(log_dens)}')
        if not isinstance(dim, int):
            raise ValueError(f'DistributionFromDensity expects integer dimension, got {type(dim)}')
        
        self._dim = dim
        self._support = support
        self._log_density = log_dens

    def log_density(self, x: ArrayLike) -> Array:
        return self._log_density(x)
    
    @property
    def dim(self):
        return self._dim

    @property
    def support(self):
        return self._support


# -----------------------------------------------------------------------------
# Light wrappers around Distribution classes for high-level Bayesian
# inverse problem API
# -----------------------------------------------------------------------------

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

LogLikelihood: TypeAlias = LogDensity

class Posterior(Distribution):
    """
    A Distribution representing the posterior distribution of a Bayesian inference
    problem, defined by specifying a Prior and Likelihood.
    """

    def __init__(self, prior: Prior, likelihood: LogLikelihood):
        if not isinstance(prior, Prior):
            raise ValueError(f'prior must be Prior, got {type(prior)}')
        if not isinstance(likelihood, LogLikelihood):
            raise ValueError(f'likelihood must be LogLikelihood/LogDensity, got {type(likelihood)}')

        self.prior = prior
        self.likelihood = likelihood

    def log_density(self, x: ArrayLike):
        return self.prior.log_density(x) + self.likelihood(x)
    
    @property
    def dim(self):
        return self.prior.dim
    
    @property
    def support(self):
        return self.prior.support
    
    def _get_log_density_function(self) -> Callable:
        """
        Returns a callable JAX-compatible log-density, suitable for passing 
        to blackbox sampling algorithms. Note that the default implmentation 
        simply returns `self.log_density`. Subclasses will often want to override 
        to apply change of variables adjustment to transform to unconstrained space.
        prior.log_density and likelihood must be jitable for the returned function
        to be jitable.
        """
        prior_logp = self.prior.log_density # bound method
        lik = self.likelihood               # callable

        def logp(x):
            return prior_logp(x) + lik(x)

        return logp

    
    def sample(self, key: PRNGKey, n: int = 1, **kwargs) -> Array:
        """ Default sampling method: NUTS """
        key_init_position, key_init_kernel, key_sample = jr.split(key, 3)
        logdensity = self._get_log_density_function()
        initial_position = self.prior.sample(key_init_position).ravel()
        initial_position = dict(zip(self.prior.par_names, initial_position))
        init_state, kernel = init_nuts_kernel(key_init_kernel, logdensity, initial_position, **kwargs)
        states = mcmc_loop(key_sample, kernel, init_state, num_samples=n)

        return stack_dict_arrays(states.position)