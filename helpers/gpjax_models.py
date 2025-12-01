# gpjax_models.py
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from flax import nnx
import numpyro.distributions.transforms as npt

from gpjax.parameters import (
    Parameter,
    transform,
    DEFAULT_BIJECTION,
)


def construct_design(design_method, n_design, prior, f):
    """
    Sample design inputs from prior, then evaluate target function f to
    construct design outputs. Return design as gpjax Dataset object.
    """

    if design_method == 'lhc':
        x_design = prior.sample_lhc(n_design)
    elif design_method == 'uniform':
        x_design = prior.sample(n_design)
    else:
        raise ValueError(f'Invalid design method {design_method}')

    x_design = jnp.asarray(x_design)
    y_design = jnp.asarray(f(x_design))

    if y_design.ndim < 2:
        y_design = y_design.reshape(-1, 1)

    return gpx.Dataset(X=x_design, y=y_design)


def construct_gp(design, set_bounds=True):

    # prior mean
    constant_param = gpx.parameters.Real(value=design.y.mean())
    meanf = gpx.mean_functions.Constant(constant_param)

    # prior kernel
    stats = _get_distance_stats_from_design(design)
    kernel = gpx.kernels.RBF(lengthscale=stats['mean'], variance=design.y.var(), n_dims=design.in_dim)

    # gp prior
    gp_prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # likelihood
    gp_likelihood = gpx.likelihoods.Gaussian(num_datapoints=design.n, obs_stddev=gp_prior.jitter)

    # gp posterior (with untuned hyperparameters)
    gp_posterior = gp_prior * gp_likelihood

    # set bound constraints based on design data
    if set_bounds:
        gp_posterior, bijection = _set_bound_constraints(gp_posterior, design)
    else:
        bijection = DEFAULT_BIJECTION

    return gp_posterior, bijection


def train_gp_hyperpars(self):
    starting_mll = -gpx.objectives.conjugate_mll(self.gp_untuned_posterior, self.design)

    gp_posterior, history = gpx.fit_scipy(
        model=self.gp_untuned_posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=self.design,
        trainable=gpx.parameters.Parameter,
    )

    ending_mll = -gpx.objectives.conjugate_mll(gp_posterior, self.design)
    opt_info = {"starting_mll": starting_mll,
                "ending_mll": ending_mll,
                "history": history}
    return gp_posterior, opt_info
    

def _set_bound_constraints(model, design):
    X_stats = _get_distance_stats_from_design(design)
    y_sd = design.y.std()
    
    param_key_to_tag = {
        'likelihood/obs_stddev': 'likelihood_std_dev',
        'prior/kernel/lengthscale': 'kernel_lengthscale',
        'prior/kernel/variance': 'kernel_variance'
    }

    # hard-coding this for now, but this should be automated with validation
    model.likelihood.obs_stddev.tag = 'likelihood_std_dev'
    model.prior.kernel.lengthscale.tag = 'kernel_lengthscale'
    model.prior.kernel.variance.tag = 'kernel_variance'

    noise_sd_low = model.prior.jitter
    noise_sd_high = jnp.maximum(0.01 * y_sd, noise_sd_low * 2)

    likelihood_std_dev_bij= _make_interval_bijector(noise_sd_low, noise_sd_high)
    kernel_lengthscale_bij = _make_interval_bijector(X_stats['min'], X_stats['max'])
    kernel_variance_bij = _make_interval_bijector((0.1 * y_sd)**2, (2 * y_sd)**2)

    # create updated bijection
    bijection = dict(DEFAULT_BIJECTION)
    bijection['likelihood_std_dev'] = likelihood_std_dev_bij
    bijection['kernel_lengthscale'] = kernel_lengthscale_bij
    bijection['kernel_variance'] = kernel_variance_bij

    return model, bijection


def _get_distance_stats_from_design(design, q=None):
    """
    Compute distances between pairwise distinct points, dimension-by-dimension.
    Return dictionary of summary statistics (mean, min, max, quantiles) summarizing
    the distribution over these univariate distances (dim-by-dim).
    """

    if q is None:
        q = jnp.linspace(0.1, 0.9, 9)

    X = design.X
    n, d = X.shape

    # shape (n, n, d); dists[i,j,k] = |x_{ik} - x_{jk}|
    dists = jnp.abs(X[:,jnp.newaxis,:] - X[jnp.newaxis,:,:])

    # only keep distinct pairs where i != j
    mask = ~jnp.eye(n, dtype=bool).ravel()

    # flatten to (n*(n-1), d)
    dists = dists.reshape(-1, d)[mask,:]

    return {
        'mean': jnp.mean(dists, axis=0),
        'min': jnp.min(dists, axis=0),
        'max': jnp.max(dists, axis=0),
        'quantiles': jnp.quantile(dists, q, axis=0)
    }


def _make_interval_bijector(low, high):
    """Return a NumPyro ComposeTransform mapping R -> (low, high)."""
    scale = jnp.asarray(high) - jnp.asarray(low)
    return npt.ComposeTransform([npt.SigmoidTransform(), 
                                 npt.AffineTransform(jnp.asarray(low), scale)])




# TODO: TEMP
def get_gp_model(key):
    # dataset
    n = 100
    noise = 0.3

    key, subkey = jr.split(key)
    x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n,)).reshape(-1, 1)
    f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
    signal = f(x)
    y = signal + jr.normal(subkey, shape=signal.shape) * noise
    D = gpx.Dataset(X=x, y=y)

    # gp prior
    kernel = gpx.kernels.RBF()  # 1-dimensional input
    meanf = gpx.mean_functions.Zero()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    # posterior
    posterior = prior * likelihood

    return posterior, D


if __name__ == '__main__':
    key = jr.key(123)
    model, design = get_gp_model(key)

    gp, bijection = construct_gp(design, set_bounds=True)
    print(gp)
    print(bijection)