# uncprop/models/elliptic_pde/inverse_problem.py
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr

import gpjax
from gpjax.kernels.stationary import PoweredExponential

from uncprop.custom_types import Array, ArrayLike, PRNGKey
from uncprop.core.inverse_problem import Prior, LogLikelihood
from uncprop.core.surrogate import GPJaxSurrogate
from uncprop.models.elliptic_pde.pde_model import get_discrete_source, solve_pde_vmap


def generate_pde_inv_prob_rep(key: PRNGKey,
                              noise_cov_tril: Array,
                              n_kl_modes: int,
                              settings: PDESettings | None = None):
    """ Top-level function for generating instance of a VSEM inverse problem

    Generates a Posterior object representing the exact posterior corresponding
    to a VSEM inverse problem.
    """

    if settings is None:
        settings = PDESettings()

    # GP prior on log permeability field
    meanf = gpjax.mean_functions.Constant(1.0)
    kernel = PoweredExponential(lengthscale=0.3, variance=1.0, power=0.3, n_dims=1)
    gp_prior = gpjax.gps.Prior(mean_function=meanf, kernel=kernel)

    (m, Phi), _ = karhunen_loeve(gp=gp_prior,
                                 settings=settings,
                                 n_kl_modes=n_kl_modes)

    prior = PDEPrior(dim=n_kl_modes)


@dataclass
class PDESettings:
    n_grid: int = 100
    left_flux: float = -1.0
    rightbc: float = 1.0
    source_wells: Array = jnp.array([0.2, 0.4, 0.6, 0.8])
    source_strength: float = 0.8
    source_width: float = 0.05

    def __post_init__(self):
        self.xgrid = jnp.linspace(0.0, 1.0, self.n_grid)
        self.low = 0
        self.high = 1
        self.dx = (self.high - self.low) / (self.n_grid - 1)
        self.source = get_discrete_source(X=self.xgrid, 
                                          well_locations=self.source_wells, 
                                          strength=self.source_strength, 
                                          width=self.source_width)


def karhunen_loeve(gp: gpjax.gps.Prior, 
                   settings: PDESettings,
                   n_kl_modes: int):
    """
    Approximates the GP prior as a sum of basis "functions" with N(0,1) weights.

    This is not really attempting a full approximation of the KL (e.g., as could
    be done with the Nystrom method). We simply project on a dense grid and 
    compute the eigendecomposition.
    """

    # Discretized prior
    prior_grid = gp(settings.xgrid)

    # Eigendecomposition - scale to account for grid spacing
    K = settings.dx * prior_grid.covariance_matrix
    K = 0.5 * (K + K.T)
    eigvals, eigvecs = jnp.linalg.eigh(K)

    # Sort descending
    idx = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Approximation using truncated basis Phi (n_grid, D)
    Phi = eigvecs[:n_kl_modes] * jnp.sqrt(eigvals[:n_kl_modes])
    m = prior_grid.mean

    return (m, Phi), (eigvals, eigvecs)


class PDEPrior(Prior):
    """
    This represents the prior over the Karhunen-Loeve modes, which 
    is simply N(0, I).
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def support(self):
        return (-jnp.inf, jnp.inf)
    
    @property
    def par_names(self):
        return [f'u{i}' for i in range(self.dim)]
    
    def log_density(self, x: ArrayLike):
        x = jnp.atleast_2d(x)
        return -0.5 * jnp.sum(x**2, axis=1)
    
    def sample(self, key: PRNGKey, n: int = 1):
        return jr.normal(key, shape=(n, self.dim))
