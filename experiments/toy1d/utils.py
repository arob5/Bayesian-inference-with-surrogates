# utils.py
# Utility and helper functions for toy 1d example.

from jax import config
config.update('jax_enable_x64', True)
from pathlib import Path
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
import gpjax as gpx
from scipy.stats import qmc
from numpy.random import default_rng
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from numpyro.distributions import LogNormal
from jax.scipy.stats import norm

from uncprop.custom_types import Array, PRNGKey
from uncprop.utils.grid import Grid, DensityComparisonGrid, normalize_density_over_grid
from uncprop.core.inverse_problem import Prior, Posterior
from uncprop.core.distribution import Distribution, DistributionFromDensity
from uncprop.utils.gpjax_models import construct_gp
from uncprop.utils.distribution import _gaussian_log_density_tril, clipped_gaussian_mean
from uncprop.utils.other import _numpy_rng_seed_from_jax_key
from uncprop.utils.gpjax_multioutput import BatchedRBF
from uncprop.models.vsem.surrogate import _estimate_ep_grid

from uncprop.core.surrogate import (
    SurrogateDistribution,
    construct_design, 
    GPJaxSurrogate, 
    FwdModelGaussianSurrogate,
    LogDensGPSurrogate,
    LogDensClippedGPSurrogate,
)
from uncprop.utils.plot import (
    set_plot_theme, 
    smart_subplots,
    plot_gp_1d,
    plot_marginal_pred_1d,
)


# -----------------------------------------------------------------------------
# Surrogate models
# -----------------------------------------------------------------------------

@dataclass
class SurrogatePost1d:
    key: PRNGKey
    post_em: SurrogateDistribution
    post_true: Posterior
    filename_label: str
    target_label: str
    grid: Grid
    n_mc: int
    f_target: Callable
    plot_surrogate_fn: Callable
    plot_log_dens_surrogate_fn: Callable
    plot_dens_surrogate_fn: Callable
    surrogate_pred: Distribution | None = None
    
    def __post_init__(self):
        key_samp, key_ep = jr.split(self.key)
        self.surrogate_pred = self.post_em.surrogate(self.grid.flat_grid)
        self.lpost_samp = self.post_em.sample_lpost(key_samp, self.grid.flat_grid, self.n_mc)
        self.density_grid = get_density_grid(key_ep, self.post_em, self.grid, self.post_true)

    def plot_surrogate(self, **kwargs):
        fig, ax = self.plot_surrogate_fn(surrogate_pred=self.surrogate_pred,
                                         design=self.post_em.surrogate.design,
                                         f_target=self.f_target,
                                         grid=self.grid,
                                         **kwargs)
        ax.set_ylabel(self.target_label)

        return (fig, ax)

    def plot_log_dens_surrogate(self, **kwargs):
        fig, ax = self.plot_log_dens_surrogate_fn(post_em_1d=self, **kwargs)
        ax.set_ylabel(r'$\log \tilde{\pi}(u)$')
        return fig, ax
    
    def plot_dens_surrogate(self, **kwargs):
        fig, ax = self.plot_dens_surrogate_fn(post_em_1d=self, **kwargs)
        ax.set_ylabel(r'$\tilde{\pi}(u)$')
        return fig, ax


class FwdModelGaussianSurrogateGrid(FwdModelGaussianSurrogate):
    """Posterior surrogate induced by forward model surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):

        cell_area = grid.cell_area
        input_dim = self.surrogate.input_dim
        y = self.y
        noise_cov_tril = self.noise_cov_tril

        def log_dens(x):
            log_post_samp = self.sample_lpost(key, x, n=n_mc)
            return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

        return DistributionFromDensity(log_dens=log_dens, dim=input_dim)

    def sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        fwd_samp = self.surrogate(x).sample(key, n=n) # (n, n_x)
        log_prior_dens = self.log_prior(x)

        y = self.y
        noise_cov_tril = self.noise_cov_tril

        log_lik_vals = jnp.zeros(fwd_samp.shape)
        for i in range(fwd_samp.shape[1]):
            l = _gaussian_log_density_tril(y, m=fwd_samp[:,i].reshape(-1,1), L=noise_cov_tril)
            log_lik_vals = log_lik_vals.at[:,i].set(l)

        log_post_samp = log_prior_dens + log_lik_vals
        return log_post_samp


class LogDensGPSurrogateGrid(LogDensGPSurrogate):
    """Posterior surrogate induced by log-density surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):
        
            cell_area = grid.cell_area
            input_dim = self.surrogate.input_dim

            def log_dens(x):
                log_post_samp = self.surrogate(x).sample(key, n=n_mc) # (n_mc, n_x)
                return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

            return DistributionFromDensity(log_dens=log_dens, dim=input_dim)
    
    def _sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        return self.surrogate(x).sample(key, n=n) # (n, n_x)
    

class LogDensClippedGPSurrogateGrid(LogDensClippedGPSurrogate):
    """Posterior surrogate induced by clipped GP log-density surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):
        
            cell_area = grid.cell_area
            input_dim = self.surrogate.input_dim

            def log_dens(x):
                log_post_samp = self._sample_lpost(key, x, n=n_mc) # (n_mc, n_x)
                return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

            return DistributionFromDensity(log_dens=log_dens, dim=input_dim)
    
    def _sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        return self.sample_surrogate_pred(key, x, n=n) # (n, n_x)
    

def wrap_gp(gp, design, jitter):
    """Wrap single output gpjax GP as batch GPJaxSurrogate object"""
    ker = gp.prior.kernel
    batched_kernel = BatchedRBF(batch_dim=1, input_dim=ker.n_dims,
                                lengthscale=ker.lengthscale,
                                variance=ker.variance)
    batched_gp_prior = gpx.gps.Prior(mean_function=gp.prior.mean_function,
                                     kernel=batched_kernel)
    batched_gp_post = batched_gp_prior * gp.likelihood
    return GPJaxSurrogate(gp=batched_gp_post, design=design, jitter=jitter)


# -------------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------------

def calc_dist_from_samples(lpost_samp: Array,
                           transform: str | None = None,
                           interval_prob: float = 0.95):
    """
    By default, computes mean and lower/upper quantiles for a set of 
    unnormalized log-density samples. Optionally transforms samples prior
    to computing these quantities. The transform arg accepts None, 'exp'
    or 'normalize'. Normalize both exponentiates and normalized.
     
    The statistics are always returned on the log scale for consistency.
    """
    samp = lpost_samp
    q_lower = 0.5 * (1 - interval_prob)

    if transform is None:
        mean = jnp.mean(samp, axis=0)
    elif transform == 'exp':
        mean = logsumexp(samp, axis=0) - jnp.log(samp.shape[0])
    elif transform == 'normalize':
        raise NotImplementedError
    else:
        raise ValueError(f'Invalid transform: {transform}')

    # Exponential transform simply exponentiates lower/upper
    lower = jnp.quantile(samp, q=q_lower, axis=0)
    upper = jnp.quantile(samp, q=1-q_lower, axis=0)
        
    return {'mean': mean, 'lower': lower, 'upper': upper}


def plot_gp_surrogate(surrogate_pred: Distribution,
                      design: gpx.Dataset,
                      f_target: Callable,
                      grid: Grid, 
                      gp_colors: dict[str, str] | None = None, 
                      interval_prob: float = 0.95):
    """Save plot summarizing underlying GP emulator distribution
    
    Agnostic to the underlying surrogate target. `f_target` is the true target
    function that is emulated.

    Returns:
        tuple:
            tuple: (figure, axis) objects
            Distribution: surrogate predictive distribution at grid points
    """
    fig_em, ax_em = plot_gp_1d(x=grid.flat_grid.ravel(),
                               mean=surrogate_pred.mean,
                               sd=surrogate_pred.stdev,
                               points=design.X,
                               true_y=f_target(grid.flat_grid),
                               colors=gp_colors,
                               interval_prob=interval_prob)

    return fig_em, ax_em


def plot_log_dens_surrogate_fwd(post_em_1d: SurrogatePost1d,
                                interval_prob: float = 0.95,
                                gp_colors: dict[str,str] | None = None):
    assert isinstance(post_em_1d.post_em, FwdModelGaussianSurrogate)
    post_em = post_em_1d.post_em
    post_true = post_em_1d.post_true
    grid = post_em_1d.grid
    pred = post_em_1d.surrogate_pred

    # Empirical statistics from samples
    stats = calc_dist_from_samples(lpost_samp=post_em_1d.lpost_samp,
                                   interval_prob=interval_prob)

    # Can compute the mean analytically
    sigma = post_em.noise_cov_tril.item()
    log_dens_mean = norm.logpdf(post_em.y, loc=pred.mean, scale=sigma) - 0.5 * pred.variance / (sigma**2)

    # Ground truth
    lpost_true = post_true.log_density(grid.flat_grid).ravel()

    fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                    mean=log_dens_mean.ravel(),
                                    lower=stats['lower'],
                                    upper=stats['upper'],
                                    points=post_em.surrogate.design.X,
                                    true_y=lpost_true,
                                    colors=gp_colors)
    
    return fig, ax


def plot_dens_surrogate_fwd(post_em_1d: SurrogatePost1d,
                            interval_prob: float = 0.95,
                            gp_colors: dict[str,str] | None = None):
    assert isinstance(post_em_1d.post_em, FwdModelGaussianSurrogate)
    post_em = post_em_1d.post_em
    post_true = post_em_1d.post_true
    grid = post_em_1d.grid

    # Empirical statistics from samples
    stats = calc_dist_from_samples(lpost_samp=post_em_1d.lpost_samp,
                                   transform='exp',
                                   interval_prob=interval_prob)

    # Ground truth
    lpost_true = post_true.log_density(grid.flat_grid).ravel()

    fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                    mean=jnp.exp(post_em_1d.density_grid.log_dens_grid['eup']),
                                    lower=jnp.exp(stats['lower']),
                                    upper=jnp.exp(stats['upper']),
                                    points=post_em.surrogate.design.X,
                                    true_y=jnp.exp(lpost_true),
                                    colors=gp_colors)
    
    return fig, ax


def plot_lognorm_1d(x,
                    mean,
                    sd,
                    colors=None,
                    points=None,
                    true_y=None,
                    interval_prob=0.95,
                    interval_alpha=0.3):
    """
    Plot lognormal process marginals, using log-scale for y-axis.
    true_y should be on log-scale.
    """
    # log of lognormal mean
    log_mean_ln = mean + 0.5 * (sd**2)

    # log of lognormal confidence interval bounds
    z = norm.ppf(1 - (1 - interval_prob) / 2)
    log_lower = mean - z * sd
    log_upper = mean + z * sd

    fig, ax = plot_marginal_pred_1d(x=x,
                                    mean=log_mean_ln,
                                    lower=log_lower,
                                    upper=log_upper,
                                    colors=colors,
                                    points=points,
                                    true_y=true_y,
                                    interval_alpha=interval_alpha)

    # Force ticks to land on nice powers of 10
    ln_10 = jnp.log(10)
    tick_spacing = ln_10 * 100
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # Format the label to show 10^integer
    def log_formatter(x, pos):
        exponent = x / ln_10
        return f"$10^{{{int(round(exponent))}}}$"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

    return fig, ax


def get_density_grid(key: PRNGKey,
                     post_em: SurrogateDistribution,
                     grid: Grid, 
                     post_true: Posterior):
    dists = {
        'exact': post_true,
        'mean': post_em.expected_surrogate_approx(),
        'eup': post_em.expected_density_approx(),
        # 'ep': post_em.expected_normalized_density_approx(key, grid=grid)
    }

    return DensityComparisonGrid(grid=grid, distributions=dists)


def save_post_approx_plot(post_em, dens_grid, out_dir, 
                          filename_label, post_colors, post_ylim=None):
    """Save posterior approximation plot."""
    # posterior approximation plot
    fig_approx, ax_approx = dens_grid.plot(normalized=True, 
                                           log_scale=False, 
                                           points=post_em.surrogate.design.X,
                                           colors=post_colors)
    
    ax_approx.set_ylabel(r'$\pi(u)$')
    if post_ylim is not None:
        ax_approx.set_ylim(post_ylim)
    
    fig_approx.savefig(out_dir / f'post_approx_{filename_label}.png', bbox_inches='tight')


def save_norm_post_surrogate_plots(key, out_dir, post_em, post_true, grid, filename_label,
                                   n_mc=int(1e5), interval_prob=0.95, gp_colors=None):
    lpost_samp = post_em._sample_lpost(key, grid.flat_grid, n=n_mc) # (n_mc, n_x)
    trajectories, _ = normalize_density_over_grid(lpost_samp, cell_area=grid.cell_area, return_log=False)

    q_lower = 0.5 * (1 - interval_prob)
    lower = jnp.quantile(trajectories, q=q_lower, axis=0)
    upper = jnp.quantile(trajectories, q=1-q_lower, axis=0)
    mean = jnp.mean(trajectories, axis=0)
    post_true, _ = normalize_density_over_grid(post_true.log_density(grid.flat_grid),
                                               cell_area=grid.cell_area, return_log=False)

    fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                    mean=mean,
                                    lower=lower,
                                    upper=upper,
                                    points=post_em.surrogate.design.X,
                                    true_y=post_true.ravel(),
                                    colors=gp_colors)
    ax.set_xlabel('u')
    ax.set_ylabel(r'$\pi(u)$')
    fig.savefig(out_dir / f'post_norm_dist_{filename_label}.png', bbox_inches='tight')

    return lpost_samp


def save_ldens_em_plots(key, grid, f_target, post_em, post_true, out_dir, gp_colors, 
                        post_colors, interval_prob, n_mc=int(1e5), **kwargs):
    """
    Surrogate plots for log-density emulator
    """
    key, key_grid, key_samp = jr.split(key, 3)
    dens_grid = get_density_grid(key_grid, grid, post_em, post_true)
    surr = post_em.surrogate
    filename_label = 'ldensem'

    save_post_approx_plot(post_em=post_em, dens_grid=dens_grid,
                          out_dir=out_dir, filename_label=filename_label,
                          post_colors=post_colors, **kwargs)

    _, pred = save_surrogate_plot(out_dir=out_dir,
                                  post_em=post_em, 
                                  f_target=f_target, 
                                  grid=grid,
                                  gp_colors=gp_colors, 
                                  filename_label=filename_label,
                                  target_label=r'$\log \tilde{\pi}(u)$', 
                                  interval_prob=interval_prob)
    
    lpost_samp = save_norm_post_surrogate_plots(key=key_samp, 
                                                out_dir=out_dir,
                                                post_em=post_em, 
                                                post_true=post_true, 
                                                grid=grid,
                                                filename_label=filename_label,
                                                n_mc=n_mc, 
                                                interval_prob=interval_prob,
                                                gp_colors=gp_colors)

    # unnormalized posterior surrogate plot
    fig_dens, ax_dens = plot_lognorm_1d(x=grid.flat_grid.ravel(),
                                        mean=pred.mean,
                                        sd=pred.stdev,
                                        points=surr.design.X,
                                        true_y=jnp.exp(f_target(grid.flat_grid)),
                                        colors=gp_colors,
                                        interval_prob=interval_prob)
    ax_dens.set_ylabel(r'$\tilde{\pi}(u)$')
    fig_dens.savefig(out_dir / f'dens_dist_{filename_label}.png', bbox_inches='tight')


# def save_clipped_ldens_em_plots(key, grid, f_target, post_em, post_true, out_dir, gp_colors, 
#                                 post_colors, interval_prob, n_mc=int(1e5), **kwargs):
#     """
#     Surrogate plots for log-density emulator
#     """
#     key, key_grid, key_samp = jr.split(key, 3)
#     dens_grid = get_density_grid(key_grid, grid, post_em, post_true)
#     surr = post_em.surrogate
#     filename_label = 'clipem'

#     save_post_approx_plot(post_em=post_em, dens_grid=dens_grid,
#                           out_dir=out_dir, filename_label=filename_label,
#                           post_colors=post_colors, **kwargs)
    
#     lpost_samp = save_norm_post_surrogate_plots(key=key_samp, 
#                                                 out_dir=out_dir,
#                                                 post_em=post_em, 
#                                                 post_true=post_true, 
#                                                 grid=grid,
#                                                 filename_label=filename_label,
#                                                 n_mc=n_mc, 
#                                                 interval_prob=interval_prob)

#     fig_gp, ax_gp = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
#                                           mean=post_em.expected_surrogate_approx().log_density(grid.flat_grid),
#                                           lower=log_lower,
#                                           upper=log_upper,
#                                           colors=colors,
#                                           points=points,
#                                           true_y=true_y,
#                                           interval_alpha=interval_alpha)

    # unnormalized posterior surrogate plot
    # fig_dens, ax_dens = plot_lognorm_1d(x=grid.flat_grid.ravel(),
    #                                     mean=pred.mean,
    #                                     sd=pred.stdev,
    #                                     points=surr.design.X,
    #                                     true_y=jnp.exp(f_target(grid.flat_grid)),
    #                                     colors=gp_colors,
    #                                     interval_prob=interval_prob)
    # ax_dens.set_ylabel(r'$\tilde{\pi}(u)$')
    # fig_dens.savefig(out_dir / f'dens_dist_{filename_label}.png', bbox_inches='tight')


def save_fwd_em_plots(key, grid, f_target, post_em, post_true, out_dir, gp_colors, 
                      post_colors, interval_prob, n_mc=int(1e5), **kwargs):
    """
    Surrogate plots for forward model emulator
    """
    key, key_grid, key_samp = jr.split(key, 3)
    dens_grid = get_density_grid(key_grid, grid, post_em, post_true)
    surr = post_em.surrogate
    filename_label = 'fwdem'

    save_post_approx_plot(post_em=post_em, dens_grid=dens_grid,
                          out_dir=out_dir, filename_label=filename_label,
                          post_colors=post_colors, **kwargs)

    _, pred = save_surrogate_plot(out_dir=out_dir,
                                  post_em=post_em, 
                                  f_target=f_target, 
                                  grid=grid,
                                  gp_colors=gp_colors, 
                                  filename_label=filename_label,
                                  target_label=r'$G(u)$', 
                                  interval_prob=interval_prob)
    
    lpost_samp = save_norm_post_surrogate_plots(key=key_samp, 
                                                out_dir=out_dir,
                                                post_em=post_em, 
                                                post_true=post_true, 
                                                grid=grid,
                                                filename_label=filename_label,
                                                n_mc=n_mc, 
                                                interval_prob=interval_prob)
   
    # induced log unnormalized posterior density surrogate
    sigma = post_em.noise_cov_tril.item()
    log_dens_mean = norm.logpdf(post_em.y, loc=pred.mean, scale=sigma) - 0.5 * pred.variance / (sigma**2)
    q_lower = 0.5 * (1 - interval_prob)
    lpost_lower = jnp.quantile(lpost_samp, q=q_lower, axis=0)
    lpost_upper = jnp.quantile(lpost_samp, q=1-q_lower, axis=0)
    lpost_true = post_true.log_density(grid.flat_grid).ravel()

    fig_lpost, ax_lpost = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                                mean=log_dens_mean.ravel(),
                                                lower=lpost_lower,
                                                upper=lpost_upper,
                                                points=surr.design.X,
                                                true_y=lpost_true,
                                                colors=gp_colors)
    ax_lpost.set_ylabel(r'$\log \tilde{\pi}(u)$')
    fig_lpost.savefig(out_dir / f'lpost_dist_{filename_label}.png', bbox_inches='tight')

    # unnormalized posterior surrogate plot
    fig_post, ax_post = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                              mean=jnp.exp(dens_grid.log_dens_grid['eup']),
                                              lower=jnp.exp(lpost_lower),
                                              upper=jnp.exp(lpost_upper),
                                              points=surr.design.X,
                                              true_y=jnp.exp(lpost_true),
                                              colors=gp_colors)
    ax_post.set_ylabel(r'$\tilde{\pi}(u)$')
    fig_post.savefig(out_dir / f'dens_dist_{filename_label}.png', bbox_inches='tight')
