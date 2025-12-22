from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from gpjax import Dataset
from gpjax.linalg.utils import add_jitter
from gpjax.gps import ConjugatePosterior
from gpjax.likelihoods import Gaussian as GPJaxGaussianLikelihood
from numpyro.distributions import (
    Distribution as NumpyroDistribution,
    MultivariateNormal,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike
from uncprop.core.distribution import (
    Distribution, 
    DistributionFromDensity, 
    GaussianFromNumpyro,
)

from uncprop.utils.distribution import (
    clipped_gaussian_mean,
    log_clipped_lognormal_mean,
    _gaussian_log_density_tril
)

# predictive distribution of a surrogate at set of inputs
PredDist: TypeAlias = Distribution


def construct_design(key: PRNGKey,
                     design_method: str, 
                     n_design: int, 
                     prior_sampler: Callable[[PRNGKey, int], ArrayLike], 
                     f: Callable) -> Dataset:
    """ Construct design (training) data for training a surrogate

    Sample design input, then evaluate target function f to
    construct design outputs. Return design as gpjax Dataset object.
    The target function should return with a (n,) or (n, q) array 
    when evaluated at a batch of n points.
    """
    x_design = prior_sampler(key, n_design)

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
    def support(self) -> tuple[ArrayLike, ArrayLike]:
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
        """ 
        Same return type as `log_density_from_pred` but first computes
        the surrogate predictions at a set of inputs. Then pushes the 
        distribution through the log density map.
        """
        pred = self.surrogate(input)
        return self.log_density_from_pred(pred)
    
    def sample_surrogate_pred(self, key: PRNGKey, input: ArrayLike, n: int = 1) -> Array:
        """
        Sample the surrogate predictive distribution at a set of inputs.
        """
        return self.surrogate(input).sample(key, n=n)

    def sample_trajectory(self) -> Distribution:
        """ Returns a Distribution representing a trajectory of the random distribution 
        
        Since a SurrogateDistribution is a random distribution, a trajectory of a 
        SurrogateDistribution is a deterministic Distribution. Subclasses will often 
        implement this by sampling a log density trajectory and then returning a 
        DistributionFromDensity. However, there may be other cases.
        """
        raise NotImplementedError('sample_trajectory() not implemented by SurrogateDistribution object.')

    def expected_surrogate_approx(self) -> Distribution:
        """ aka 'plug-in mean' """
        raise NotImplementedError
    
    def expected_log_density_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_density_approx(self) -> Distribution:
        """ aka 'expected unnormalized posterior' """
        raise NotImplementedError
    
    def expected_normalized_density_approx(self, *args, **method_kwargs) -> Distribution:
        """ aka 'expected posterior' """
        raise NotImplementedError
    
    def _expected_normalized_imputation(self, 
                                        n_trajectory: int = 100, 
                                        n_samp_per_trajectory: int = 1):
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Log-Density Surrogates
# -----------------------------------------------------------------------------

class LogDensGPSurrogate(SurrogateDistribution):
    """
    A SurrogateDistribution defined by a Gaussian log-density surrogate.
    Concretely, the random distribution is:
        pi(u; f) = exp{f(u)} / Z(f), Z(f) = int_u exp{f(u)} du, where f(u) ~ N(m(u), C(u))

    That is, f is a Gaussian process.
    """

    def __init__(self, 
                 log_dens: GPJaxSurrogate, 
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        if not isinstance(log_dens, GPJaxSurrogate):
            raise ValueError(f'LogDensGPSurrogate requires `log_dens` to be a GPJaxSurrogate, got {type(log_dens)}')
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

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

        gp = self.surrogate
        surrogate_mean = lambda x: gp.mean(x)

        return DistributionFromDensity(log_dens=surrogate_mean,
                                       dim=gp.input_dim)
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        """ Surrogate is the log-density, so equivalent to expected_surrogate_approx. """
        return self.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        """ Density surrogate exp{f(u)} is log-normal, so expectation is log-normal mean """
        gp = self.surrogate

        def log_expected_dens(x):
            pred = gp(x)
            return pred.mean + 0.5 * pred.variance
        
        return DistributionFromDensity(log_dens=log_expected_dens,
                                       dim=gp.input_dim)
    

class LogDensClippedGPSurrogate(SurrogateDistribution):
    """
    The same form as LogDensGPSurrogate, but the GP f(u) is "clipped"; i.e., it
    is replaced by g(u) := min{f(u), b(u)} where b(u) is a given pointwise upper
    bound. This is useful when the posterior density being approximated has a 
    known upper bound.

    The GP f(u) is still considered the "surrogate" here (`self.surrogate`); the clipping
    transformation is simply applied post-hoc to any predictive quantities.
    """

    def __init__(self, 
                 log_dens: GPJaxSurrogate,
                 log_dens_upper_bound: Callable[[ArrayLike], Array],
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        if not isinstance(log_dens, GPJaxSurrogate):
            raise ValueError(f'LogDensGPSurrogate requires `log_dens` to be a GPJaxSurrogate, got {type(log_dens)}')
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._surrogate = log_dens
        self._log_dens_upper_bound = log_dens_upper_bound
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
    
    def sample_surrogate_pred(self, key: PRNGKey, input: ArrayLike, n: int = 1) -> Array:
        """
        Clip the Gaussian samples
        """
        gaussian_samp = self.surrogate(input).sample(key, n=n)
        upper_bound = self._log_dens_upper_bound(input).ravel()
        return jnp.clip(gaussian_samp, max=upper_bound)
    
    def log_density_from_pred(self, pred: PredDist):
        """ Surrogate predictions are Gaussian log-density predictions; just clip them """
        return pred
    
    def expected_surrogate_approx(self) -> DistributionFromDensity:
        """ Plug-in surrogate mean as log-density approximation. """
        log_dens_upper_bound = self._log_dens_upper_bound
        gp = self.surrogate

        def surrogate_mean(x):
            gaussian_mean = gp.mean(x)
            gaussian_var = gp.variance(x)
            upper_bound = log_dens_upper_bound(x)
            return clipped_gaussian_mean(gaussian_mean, gaussian_var, upper_bound)

        return DistributionFromDensity(log_dens=surrogate_mean,
                                       dim=gp.input_dim)
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        """ Surrogate is the log-density, so equivalent to expected_surrogate_approx. """
        return self.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        """ Density surrogate exp{f(u)} is log-normal, so expectation is clipped log-normal mean """
        log_dens_upper_bound = self._log_dens_upper_bound
        gp = self.surrogate

        def log_expected_dens(x):
            gaussian_mean = gp.mean(x)
            gaussian_var = gp.variance(x)
            upper_bound = log_dens_upper_bound(x)
            return log_clipped_lognormal_mean(gaussian_mean, gaussian_var, upper_bound)
        
        return DistributionFromDensity(log_dens=log_expected_dens,
                                       dim=gp.input_dim)


# -----------------------------------------------------------------------------
# Forward Model Surrogates
# -----------------------------------------------------------------------------

class FwdModelGaussianSurrogate(SurrogateDistribution):
    """
    A random distribution induced by a (potentially multi-output) GP pushed 
    through a Gaussian likelihood.

    Assumes the model:
        pi(u; f) = pi_0(u) * N(y | f(u), C), f ~ GP(m, k)

    If y is an (p,) array, then evaluating f(U) on a batch of m inputs should
    produce a GaussianFromNumpyro with event shape (m,) and batch shape (p,).
    This implies the mean/variance will be (p,m) and covariance/cholesky
    will be (p,m,m).

    Notes:
        If the multi-output GP models correlation across outputs, at present
        this correlation is not used in this class - for now treats all 
        GPs as batch independent multioutput models.
    """

    def __init__(self,
                 gp: GPJaxSurrogate,
                 log_prior: Callable,
                 y: Array,
                 noise_cov_tril: Array,
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._support = support
        self._surrogate = gp
        self.log_prior = log_prior
        self.y = y
        self.noise_cov_tril = noise_cov_tril

    @property
    def surrogate(self) -> Surrogate:
        return self._surrogate

    @property
    def dim(self) -> int:
        return self.surrogate.input_dim

    @property
    def support(self) -> tuple[ArrayLike, ArrayLike]:
        return self._support
    
    @property
    def noise_cov(self):
        return self.noise_cov_tril @ self.noise_cov_tril.T

    def expected_surrogate_approx(self) -> Distribution:
        gp = self.surrogate
        log_prior = self.log_prior
        y = self.y
        noise_cov_tril = self.noise_cov_tril

        def logdensity(x):
            m_x = gp(x).mean.T
            log_prior_x = log_prior(x)
            return log_prior_x + _gaussian_log_density_tril(y, m=m_x, L=noise_cov_tril)

        return DistributionFromDensity(log_dens=logdensity,
                                       dim=gp.input_dim,
                                       support=self.support)
    
    
    def expected_density_approx(self) -> Distribution:
        """
        The expected likelihood is:

            E[N(y | f(u), C)] = N(y | m(u), C + k(u))

        Notes:
            The Cholesky factor for C + k(u) can be updated more efficiently using
            rank one Cholesky updates, but for simplicity the full Cholesky is 
            recomputed for now.
        """
        gp = self.surrogate
        log_prior = self.log_prior
        y = self.y
        noise_cov = self.noise_cov

        def logdensity(x):
            pred = gp(x)
            var_x = pred.variance.T # (n, p)
            log_prior_x = log_prior(x)
            C_x = noise_cov[None] + jax.vmap(jnp.diag)(var_x) # (n, p, p)
            L_x = jnp.linalg.cholesky(C_x, upper=False)
            return log_prior_x + _gaussian_log_density_tril(y, m=pred.mean.T, L=L_x)

        return DistributionFromDensity(log_dens=logdensity,
                                       dim=gp.input_dim,
                                       support=self.support)


# -----------------------------------------------------------------------------
# gpjax wrapper used for all GP surrogates
# -----------------------------------------------------------------------------


class GPJaxSurrogate(Surrogate):
    """
    Wrapper around a gpjax conjugate posterior object. The primary reasons
    for this wrapper are to:
    (1) automatically wrap gpjax Gaussian predictions as GaussianFromNumpyro objects,
        so they are valid Distributions.
    (2) implement an update method for fast GP conditioning at new points.
    (3) support vectorized prediction for independent multioutput GPs.

    The `condition_then_predict()` method is written to return the conditional predictions
    without altering the current GP. This design is motivated by the primary use case in 
    the rk-pcn algorithm, which does not require repeatedly conditioning the GP. A copy
    method can easily be added to return an updated conditional GPJaxSurrogate when this
    functionality is required.

    Notes:
        - `jitter` is a value added to the diagonal of any predictive covariance. Note that 
        the gpjax jitter is added regardless; this provides an opportunity to increase the 
        jitter if necessary.
        - At present, the conditioning functionality is only implemented for single output
          GPs. The other methods work for both single and multioutput models. 
    """
    def __init__(self, 
                 gp: ConjugatePosterior, 
                 design: Dataset, 
                 jitter: float = 0.0):
        assert isinstance(gp, ConjugatePosterior)
        assert isinstance(design, Dataset)
        assert isinstance(gp.likelihood, GPJaxGaussianLikelihood)

        self.jitter = jitter
        self.gp = gp
        self.design = design
        self.sig2_obs = jnp.square(self.gp.likelihood.obs_stddev.get_value())
        self.P = self._compute_kernel_precision()

    @property
    def input_dim(self):
        return self.design.in_dim
    
    @property
    def output_dim(self):
        return self.design.y.shape[1]
    
    def __call__(self, input: ArrayLike) -> GaussianFromNumpyro:
        return self.predict(input)
    
    def prior_gram(self, x: Array):
        """
        Exposes the gram() method of the prior kernel of the underlying GP.
        Wraps so that the result always has a batch dimension in the 
        first dimension. In addition converts to dense, rather than returning
        a linear operator.
        """
        gram = self.gp.prior.kernel.gram(x).to_dense()

        if self.output_dim == 1:
            return self._promote_to_batch(gram)
        else:
            return self._flip(gram)

    def prior_cross_covariance(self, x: Array, z: Array):
        """
        Same as prior_gram() except for cross covariance.
        """
        cross_cov = self.gp.prior.kernel.cross_covariance(x, z)

        if self.output_dim == 1:
            return self._promote_to_batch(cross_cov)
        else:
            return self._flip(cross_cov)

    def predict(self, input: ArrayLike) -> GaussianFromNumpyro:
        return self._predict_using_precision(input, self.P, self.design)

    def condition_then_predict(self, 
                               input: ArrayLike,
                               given: tuple[Array, Array]):
        """
        Condition on new design points, then predict using conditioned GP.
        """        
        # Not yet generalized to batch setting
        assert self.output_dim == 1

        Sigma_inv, design = self._update_kernel_precision(given)
        return self._predict_using_precision(input, Sigma_inv, design)


    def _predict_using_precision(self, 
                                 x: ArrayLike, 
                                 P: Array, 
                                 design: Dataset) -> GaussianFromNumpyro:
        """ 
        Compute posterior GP multivariate normal prediction at test inputs `x`, conditional
        on dataset `design`. `P` is the inverse of the kernel matrix (including the noise covariance)
        evaluated at `design.X`.

        Predictions include the observation noise.
        """
        x = jnp.asarray(x).reshape(-1, self.input_dim)
        X, Y = design.X, design.y
        n, q = X.shape[0], self.output_dim
        m = x.shape[0]
        meanf = self.gp.prior.mean_function

        # Jitter matrices
        JX = self._get_jitter_matrix(n)
        Jx = self._get_jitter_matrix(m)

        # prior means - gpjax mean always returns (n, q)
        mx = self._flip(meanf(x))
        mX = self._flip(meanf(X))

        # prior covariances
        kX = self.prior_gram(X) + JX
        kx = self.prior_gram(x)

        kxX_P, kxX = self._compute_kxX_P(x, P, design)

        # conditional mean and covariance
        m_pred = mx[..., None] + kxX_P @ (self._flip(Y) - mX)[..., None]
        m_pred = m_pred.squeeze(-1)
        k_pred = kx + Jx - kxX_P @ jnp.transpose(kxX, axes=(0, 2, 1))

        if q == 1:
            m_pred = m_pred.squeeze(0)
            k_pred = k_pred.squeeze(0)

        gaussian_pred = MultivariateNormal(m_pred, k_pred)
        return GaussianFromNumpyro(gaussian_pred)


    def _compute_kxX_P(self,
                       x: ArrayLike, 
                       P: Array, 
                       design: Dataset):
        """
        Computes k(x, X) @ P, where X are the training inputs and
        x are m test inputs. P is the inverse kernel matrix k(X)^{-1}.

        Returns:
            tuple, with:
                kxX_P: (q, m, n)
                kxX: (q, m, n)
        """
        ker = self.gp.prior.kernel
        X = design.X
        q = self.output_dim

        kxX = self.prior_cross_covariance(x, X)
        kxX_P = kxX @ P # (q, m, n)

        return kxX_P, kxX


    def _update_kernel_precision(self, given: tuple[ArrayLike, ArrayLike]):
        """ 
        Updates the inverse kernel matrix Sigma_inv = (K + sig2*I)^{-1} when a 
        batch of new conditioning points Xnew are added. Returns the updated inverse 
        kernel matrix as well as the updated dataset (union of the old dataset and 
        the newly added points). `given` is either a gpjax `Dataset` containing the 
        new input/output pairs or a tuple (Xnew, ynew) containing the same information.
        """

        # Not yet generalized to batch setting
        assert self.output_dim == 1

        Xnew = given[0].reshape(-1, self.input_dim)
        ynew = given[1].reshape(-1, 1)
        new_dataset = self.design + Dataset(X=Xnew, y=ynew)
        
        # Partitioned matrix inverse updates.
        num_new_points = Xnew.shape[0]
        Sigma_inv = self.P.squeeze(0) # since we are assuming q=1 here
        Xcurr = self.design.X.copy()

        for i in range(num_new_points):
            xnew = Xnew[i]
            Sigma_inv = self._update_single_point_kernel_precision(Sigma_inv, Xcurr, xnew)
            Xcurr = jnp.vstack([Xcurr, xnew])

        # bring back batch dimension
        return Sigma_inv[None, ...], new_dataset
    

    def _update_single_point_kernel_precision(self, Sigma_inv, X, xnew):
        """ 
        Updates the inverse kernel matrix Sigma_inv = (K + sig2*I)^{-1} when a 
        single conditioning point xnew is added. X is the current conditioning 
        set and xnew is the new point to add.
        """

        # Not yet generalized to batch setting
        assert self.output_dim == 1

        xnew = xnew.reshape(1, -1) # (1, 1)
        knm = self.gp.prior.kernel.cross_covariance(X, xnew) # (n, 1)
        Kinv_knm = Sigma_inv @ knm # (n, 1)
        k_new_new = self.gp.prior.kernel.gram(xnew).to_dense().squeeze() # scalar
        kappa = k_new_new + self.sig2_obs + self.gp.jitter + self.jitter - (knm.T @ Kinv_knm).squeeze() # scalar
        
        outer = Kinv_knm @ Kinv_knm.T  # (n,1) @ (1,n) -> (n,n)
        top_left= Sigma_inv + outer / kappa # (n,n)
        top_right = -Kinv_knm / kappa
        bottom_left = top_right.T
        bottom_right = jnp.array([[1.0 / kappa]]) # (1,1)

        top = jnp.hstack([top_left, top_right])       # shape (n, n+1)
        bottom = jnp.hstack([bottom_left, bottom_right])  # shape (1, n+1)

        return jnp.vstack([top, bottom]) # (n+1, n+1)


    def _compute_kernel_precision(self):
        """ Inverse of the kernel matrix evaluated at the design points in self.design.X 
        
        Includes the observation noise covariance. 
        """
        n = self.design.n
        X = self.design.X
        J = self._get_jitter_matrix(n)
        ker = self.gp.prior.kernel

        # Cholesky factor of kernel matrix
        K = self._flip(ker.gram(X).to_dense()) + J
        K = K.reshape(self.output_dim, n, n)
        L = jnp.linalg.cholesky(K, upper=False)

        # Invert kernel matrix
        I = jnp.eye(n)
        I = jnp.broadcast_to(I, (self.output_dim, n, n))
        P = cho_solve((L, True), I)

        return P


    def _get_jitter_matrix(self, size: int):
        q = self.output_dim

        eye = jnp.eye(size)[None, :, :]
        eye = jnp.broadcast_to(eye, (q, size, size))

        composite_jitter = self.sig2_obs + self.gp.jitter + self.jitter
        composite_jitter = jnp.broadcast_to(composite_jitter, (q,))

        J = composite_jitter[:, None, None] * eye # (q, size, size), zeros everywhere except diagonal
        return J
    
    @staticmethod
    def _flip(a):
        """Move last axis to the first position"""
        return jnp.moveaxis(a, -1, 0)
    
    @staticmethod
    def _promote_to_batch(a):
        return a[None, ...]