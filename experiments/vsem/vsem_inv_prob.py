# experiments/test/linear_Gaussian/LinGaussTest.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, List, Optional, Tuple
from numpy.typing import NDArray
from typing import Protocol
from scipy.stats import uniform
from math import ceil

import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
from flax import nnx

import vsem_jax as vsem

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import (
    MarkovKernel, 
    GaussMetropolisKernel, 
    DiscretePCNKernel, 
    UncalibratedDiscretePCNKernel, 
    mvn_logpdf
)

import sys
sys.path.append("./../linear_Gaussian/")
from Gaussian import Gaussian

sys.path.append("./../../helpers/")
from rectified_gaussian import RectifiedGaussian


Array = NDArray

class Prior(Protocol):
    rng: np.random.Generator

    @property
    def dim(self) -> int:
        ...

    def sample(self, n: int = 1) -> Array:
        """ Out: (n,d)"""
        ...

    def log_density(self, x: Array) -> Array:
        """ In: (n,d), Out: (n,)"""
        ...

class Likelihood(Protocol):
    def log_density(self, x: Array) -> Array:
        """ In: (n,d), Out: (n,)"""
        ...


def _uniform(lower, upper):
    """
    Wrapper for scipy.stats.Uniform that parameterizes in terms of lower and
    upper bounds instead of (loc, scale).
    """
    return uniform(loc=lower, scale=upper-lower)


class VSEMPrior:
        _dists = {
            "kext": _uniform(0.4, 1.0), # _uniform(0.2, 1.0),
            "lar" : _uniform(0.2, 3.0),
            "lue" : _uniform(5e-04, 4e-03),
            "gamma": _uniform(2e-01, 6e-01),
            "tauv": _uniform(5e+02, 3e+03),
            "taus": _uniform(4e+03, 5e+04),
            "taur": _uniform(5e+02, 3e+03), 
            "av": _uniform(0.4, 1.0), # _uniform(2e-01, 1.0),
            "veg_init": _uniform(0.0, 10.0), 
            "soil_init": _uniform(0.0, 30.0),
            "root_init": _uniform(0.0, 10.0)
        }

        def __init__(self, par_names=None, rng=None):
            self.rng = rng or np.random.default_rng()
            self._par_names = par_names or vsem.get_vsem_par_names()

        @property
        def dists(self):
            return {par: self._dists[par] for par in self._par_names}

        @property
        def dim(self):
            return len(self._par_names)
        
        @property
        def par_names(self):
            return self._par_names
        
        def sample(self, n=1):
            samp = np.empty((n, self.dim))
            for j in range(self.dim):
                par_name = self._par_names[j]
                samp[:,j] = self.dists[par_name].rvs(n, random_state=self.rng)

            return samp[0] if n == 1 else samp

        def log_density(self, x):
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)

            log_dens = np.zeros(x.shape[0])
            for par_idx, par_name in enumerate(self._par_names):
                log_dens = log_dens + self.dists[par_name].logpdf(x[:,par_idx])
            
            return log_dens
        
        def simulate_ground_truth(self):
            """
            Return dictionary representing a sample from all of the parameters, 
            not just the calibration parameters.
            """
            par = {}
            for par_name, prior in self._dists.items():
                par[par_name] = prior.rvs(1, random_state=self.rng)[0]

            return par


class VSEMLikelihood:

    def __init__(self, rng, n_days, par_names, ground_truth=None):
        """
        If provided, `ground_truth` is a dictionary of all VSEM parameters that 
        will be treated as the ground truth. 
        """
        self.rng = rng
        self.n_days = n_days
        self.time_steps, self.driver = vsem.get_vsem_driver(self.n_days, self.rng)
        self.par_names = par_names
        self.d = len(par_names)

        all_par_names = vsem.get_vsem_par_names()
        self._par_idx = [all_par_names.index(par) for par in self.par_names]

        # Observation operator defined as monthly averages of LAI.
        self.lai_idx = vsem.get_vsem_output_names().index("lai")
        self.month_start_idx = np.arange(start=0, stop=self.n_days, step=31)
        month_stop_idx = np.empty_like(self.month_start_idx)
        month_stop_idx[:-1] = self.month_start_idx[1:]
        month_stop_idx[-1] = self.n_days - 1
        self.month_stop_idx = month_stop_idx
        self.month_midpoints = np.round(0.5 * (self.month_start_idx + self.month_stop_idx))

        # Ground truth
        if ground_truth is None:
            self._all_par_true = {
                "kext": 0.85, # 7.92301322e-01,
                "lar": 1.86523322e+00,
                "lue": 6.84170991e-04,
                "gamma": 5.04967614e-01,
                "tauv": 2.95868049e+03,
                "taus": 2.58846896e+04,
                "taur": 1.77011520e+03,
                "av": 0.85, # 6.88359631e-01,
                "veg_init": 3.04573410e+00,
                "soil_init": 2.11415896e+01,
                "root_init": 5.58376223e+00
            }
        else:
            self._all_par_true = ground_truth

        self.par_true = np.array([self._all_par_true[par] for par in self.par_names])
        
        # VSEM forward model.
        self.forward_model = vsem.build_vectorized_partial_forward_model(self.driver, self.par_names,
                                                                         par_default=self._all_par_true)

        # self.par_true = vsem.DefaultVSEMPrior(rng=self.rng).sample().flatten()
        self.vsem_output_true = self.forward_model(self.par_true)
        self.observable_true = self.obs_op(self.vsem_output_true).flatten()
        self.n = self.observable_true.size
        self._sigma = 0.1 * np.std(self.observable_true)
        self.noise = Gaussian(cov=self._sigma * np.identity(self.n))
        self.y = self.observable_true + self.noise.sample()
        self._likelihood_rv = Gaussian(mean=self.y, cov=self.noise.cov)
        
    def plot_driver(self):
        plt.plot(self.time_steps, self.driver, "o")
        plt.xlabel("days")
        plt.ylabel("PAR")
        plt.show()

    def par_to_obs_map(self, par):
        vsem_output = self.forward_model(par)
        return self.obs_op(vsem_output)

    def obs_op(self, vsem_output):
        """ Observation operator: monthly averages of LAI """
        lai_output = vsem_output[:,:,self.lai_idx]
        monthly_lai_averages = np.array(
            [lai_output[:, start:end].mean(axis=1) for start, end in zip(self.month_start_idx, self.month_stop_idx)]
        )        

        return monthly_lai_averages.T
    
    def log_density(self, x):
        pred_obs = self.par_to_obs_map(x)
        return self._likelihood_rv.log_p(pred_obs)

    def log_density_upper_bound(self, x):
        return self._likelihood_rv.logdet

    def plot_vsem_outputs(self, par, burn_in_start=0, include_predicted_obs=False):
        output = self.forward_model(par)
        fig, axs = vsem.plot_vsem_outputs(output[:,burn_in_start:,:], nrows=2)

        if include_predicted_obs:
            pred_obs = self.obs_op(output)
            axs[self.lai_idx].plot(self.month_midpoints, pred_obs.T, "o", color="red")

        return fig
    
    def plot_ground_truth(self):
        fig, axs = vsem.plot_vsem_outputs(self.vsem_output_true, nrows=2)

        lai_ax = axs[self.lai_idx]
        lai_ax.plot(self.month_midpoints, self.y, "o", color="red")

        return fig



class InvProb:

    def __init__(self, rng, prior: Prior, likelihood: Likelihood, sampler=None, **sampler_kwargs):
        self.rng = rng
        self.prior = prior
        self.likelihood = likelihood
        self.dim = int(self.prior.dim)
        self.par_names = prior.par_names
        self.sampler = self._set_default_sampler(**sampler_kwargs) if sampler is None else sampler
        self.posterior = None 

    def log_posterior_density(self, x: Array) -> Array:
        return self.prior.log_density(x) + self.likelihood.log_density(x)
    
    def _set_default_sampler(self, proposal_cov: Array | None = None, **kwargs):
        """ Defaults to Metropolis-Hastings. """
        
        if proposal_cov is None:
            proposal_cov = np.identity(self.dim)

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample()})

        # Target density.
        post_log_dens = lambda state: self.log_posterior_density(state.primary["u"])
        target = TargetDensity(LogDensityTerm("post", post_log_dens))

        # Metropolis-Hastings kernel with Gaussian proposal.
        mh_kernel = GaussMetropolisKernel(target, proposal_cov=proposal_cov, rng=self.rng)

        # Sampler
        sampler = BlockMCMCSampler(target, initial_state=state, kernels=mh_kernel, rng=self.rng)
        return sampler
    

    def sample_posterior(self, n_step: int, burn_in_start: int | None = None, 
                         sampler_kwargs=None, plot_kwargs=None):
        """
        Runs MCMC sampler, collects samples, and drops burn-in. Returns
        `n_samp` samples after burn-in. Default burn-in is to take second
        half of samples.
        """
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        self.sampler.sample(num_steps=n_step, **sampler_kwargs)

        # Store samples in array.
        burn_in_start = burn_in_start or round(n_step / 2)
        itr_range = np.arange(burn_in_start, len(self.sampler.trace))
        n_samp = len(itr_range)
        samp = np.empty((n_samp, self.dim))

        for samp_idx, trace_idx in enumerate(itr_range):
            samp[samp_idx,:] = self.sampler.trace[trace_idx].primary["u"]

        return samp, self.get_trace_plot(samp, **plot_kwargs)

    def reset_sampler(self):
        self.sampler.reset()

    def get_trace_plot(self, samp, nrows=1, ncols=None, col_labs=None, figsize=(5,4), plot_kwargs=None):
        n_itr, n_cols = samp.shape
        x = np.arange(n_itr)

        if plot_kwargs is None:
            plot_kwargs = {}

        if ncols is None:
            ncols = int(np.ceil(n_cols / nrows))

        if col_labs is None:
            col_labs = self.par_names

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for col in range(n_cols):
            ax = axs[col]
            ax.plot(x, samp[:,col], **plot_kwargs)
            ax.set_title(col_labs[col])
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Value")

        # Hide unused axes and close figure.
        for k in range(n_cols, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)

        return fig


class VSEMTest:
    """ Uncertainty propagation experiment for surrogate modeling for VSEM inverse problem.

    Note that this class is specialized for an inverse problem with a 2d input space.
    """
    
    def __init__(self, inv_prob: InvProb, n_design: int, 
                 n_test_grid_1d: int = 50, store_pred_rect=False):
        self.inv_prob = inv_prob
        self.n_design = n_design
        self.set_test_grid_info(n_test_grid_1d)
        self.set_gp_model()
        self.store_gp_pred(store_pred_rect)

    def set_gp_model(self):
        self.design = self.construct_design()
        self.gp_prior = self.construct_gp_prior()
        self.gp_likelihood = self.construct_gp_likelihood()
        self.gp_untuned_posterior = self.gp_prior * self.gp_likelihood
        self.gp_posterior, self.opt_info = self.train_gp_hyperpars()

    def set_test_grid_info(self, n_test_grid_1d):
        """
        Note that this method currently assumes 2d bounded input space.
        """
        par_names = self.inv_prob.par_names
        u1_supp = self.inv_prob.prior.dists[par_names[0]].support()
        u2_supp = self.inv_prob.prior.dists[par_names[1]].support()

        u1_grid = np.linspace(u1_supp[0], u1_supp[1], n_test_grid_1d)
        u2_grid = np.linspace(u2_supp[0], u2_supp[1], n_test_grid_1d)

        U1_grid, U2_grid = np.meshgrid(u1_grid, u2_grid, indexing='xy')
        U_grid= np.stack([U1_grid.ravel(), U2_grid.ravel()], axis=1)
        log_post = self.inv_prob.log_posterior_density(U_grid)
        log_post_grid = log_post.reshape(U1_grid.shape)

        self.test_grid_info = {
            "u1_grid": u1_grid,
            "u1_grid": u2_grid,
            "U1_grid": U1_grid,
            "U2_grid": U2_grid,
            "U_grid": U_grid,
            "log_post": log_post,
            "log_post_grid": log_post_grid,
            "axis_labels": par_names,
            "n_grid_1d": n_test_grid_1d,
            "n_grid": U_grid.shape[0]
        }

    def store_gp_pred(self, store_pred_rect=False):
        """
        Storing predictions as Gaussian distributions. Wrapping in custom
        Gaussian class. Optionally store rectified (clipped) Gaussian
        predictions as well.
        """
        U = self.test_grid_info["U_grid"]
        upper_bound = self.log_post_upper_bound(U)

        # Prior predictions
        # prior_latent = self.gp_posterior.prior.predict(U)
        # prior_pred = self.gp_posterior.likelihood(prior_latent)
        
        # Conditional predictions
        post_latent = self.gp_posterior.predict(U, train_data=self.design)
        post_pred = self.gp_posterior.likelihood(post_latent)
        
        # Optional store rectified Gaussian predictions
        if store_pred_rect:
            # prior_pred_rect = RectifiedGaussian(prior_pred.mean, 
            #                                     prior_pred.covariance_matrix,
            #                                     upper=upper_bound,
            #                                     rng=self.inv_prob.rng)
            post_pred_rect = RectifiedGaussian(post_pred.mean, 
                                               post_pred.covariance_matrix,
                                               upper=upper_bound,
                                               rng=self.inv_prob.rng)
        else:
            prior_pred_rect = None
            post_pred_rect = None

        # Wrap as `Gaussian`
        # prior_latent = Gaussian(prior_latent.mean, prior_latent.covariance_matrix, rng=self.inv_prob.rng)
        # prior_pred = Gaussian(prior_pred.mean, prior_pred.covariance_matrix, rng=self.inv_prob.rng)
        post_latent = Gaussian(post_latent.mean, post_latent.covariance_matrix, rng=self.inv_prob.rng)
        post_pred = Gaussian(post_pred.mean, post_pred.covariance_matrix, rng=self.inv_prob.rng)

        # self.gp_prior_pred = {"latent": prior_latent, "pred": prior_pred, "pred_rect": prior_pred_rect}
        self.gp_post_pred = {"latent": post_latent, "pred": post_pred, "pred_rect": post_pred_rect}


    def construct_design(self):
        x_design = jnp.asarray(self.inv_prob.prior.sample(self.n_design))
        y_design = jnp.asarray(self.inv_prob.log_posterior_density(x_design)).reshape((-1,1))
        return gpx.Dataset(X=x_design, y=y_design)

    def construct_gp_mean(self):
        constant_param = gpx.parameters.Real(value=self.design.y.mean())
        meanf = gpx.mean_functions.Constant(constant_param)
        return meanf

    def construct_gp_kernel(self):
        lengthscales_init = average_pairwise_distance_per_dim(self.design.X)
        lengthscales_init = jnp.array(lengthscales_init)

        ker_var_init = gpx.parameters.PositiveReal(self.design.y.var())
        kernel = gpx.kernels.RBF(lengthscale=lengthscales_init, variance=ker_var_init, n_dims=self.inv_prob.dim)
        return kernel

    def construct_gp_prior(self):
        meanf = self.construct_gp_mean()
        kernel = self.construct_gp_kernel()
        gp_prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        return gp_prior
    
    def construct_gp_likelihood(self):
        obs_stddev = gpx.parameters.PositiveReal(self.gp_prior.jitter)
        gp_likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.design.n, obs_stddev=obs_stddev)
        return gp_likelihood

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
    
    def predict(self, u, pred=None, rectify=True):
        """ Return Gaussian representing predictions (including observation noise) at u """
        if pred is None:
            latent = self.gp_posterior.predict(u, train_data=self.design)
            pred = self.gp_posterior.likelihood(latent)

            if rectify:
                pred = RectifiedGaussian(pred.mean, pred.covariance_matrix,
                                         upper=self.log_post_upper_bound(u),
                                         rng=self.inv_prob.rng)
            else:
                pred = Gaussian(pred.mean, pred.covariance_matrix, rng=self.inv_prob.rng)

        return pred

    def log_post_upper_bound(self, u: Array) -> Array:
        return self.inv_prob.likelihood.log_density_upper_bound(u) + self.inv_prob.prior.log_density(u)

    def log_post_approx_mean(self, u, pred=None, rectify=True):
        """ Log of plug-in mean approximation of unnormalized posterior density. """
        pred = self.predict(u, pred, rectify=rectify)
        return pred.mean

    def log_post_approx_eup(self, u, pred=None, rectify=True):
        """ Log of EUP approximation of unnormalized posterior density. """
        pred = self.predict(u, pred)
        return pred.mean + 0.5 * pred.variance ** 2 

    def log_post_approx_ep(self, u, n_mc=10000, pred=None, rectify=True):
        """ Log of EP approximation of normalized posterior density (approximate).
        The grid of test points is used to approximate the normalizing constants Z(f).
        The expectation with respect to f is estimated via simple Monte Carlo, by 
        sampling discretizations of f at the test points. `n_mc` is the number of 
        Monte Carlo samples to use.
        """
        pred = self.predict(u, pred)
        n_grid = pred.dim

        # log_post_samp[i,j] = log pi(u_j; f_i)
        log_post_samp = pred.sample(n_mc) # (n_mc, n_grid)

        _, log_ep_approx, _ = estimate_ep_grid(log_post_samp)
        return log_ep_approx
    
    def _exact_post_grid(self, shape_to_grid=True, log_scale=False):
        """ Normalized exact posterior density evaluated at test grid """
        U = self.test_grid_info['U_grid']
        unnormalized_post = self.test_grid_info['log_post']
        normalized_post = _normalize_over_grid(unnormalized_post, log_scale=log_scale).flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            return normalized_post.reshape(n_grid_1d, n_grid_1d)
        else:
            return normalized_post

    def _mean_approx_grid(self, shape_to_grid=True, log_scale=False, pred_type='pred'):
        """ Normalized plug-in mean approximation evaluated at test grid 
        
        `pred_type` is either 'latent', 'pred', or 'pred_rect'
        """
        U = self.test_grid_info['U_grid']
        pred = self.gp_post_pred[pred_type]
        unnormalized_approx = self.log_post_approx_mean(U, pred=pred) # (n_grid,)
        normalized_approx = _normalize_over_grid(unnormalized_approx, log_scale=log_scale).flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            return normalized_approx.reshape(n_grid_1d, n_grid_1d)
        else:
            return normalized_approx

    def _eup_approx_grid(self, shape_to_grid=True, log_scale=False, pred_type='pred'):
        """ Normalized EUP approximation evaluated at test grid """
        U = self.test_grid_info['U_grid']
        pred = self.gp_post_pred[pred_type]
        unnormalized_approx = self.log_post_approx_eup(U, pred=pred) # (n_grid,)
        normalized_approx = _normalize_over_grid(unnormalized_approx, log_scale=log_scale).flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            return normalized_approx.reshape(n_grid_1d, n_grid_1d)
        else:
            return normalized_approx

    def _ep_approx_grid(self, shape_to_grid=True, log_scale=False, pred_type='pred'):
        """ Normalized EP approximation evaluated at test grid """
        U = self.test_grid_info['U_grid']
        pred = self.gp_post_pred[pred_type]
        unnormalized_approx = self.log_post_approx_ep(U, pred=pred) # (n_grid,)
        normalized_approx = _normalize_over_grid(unnormalized_approx, log_scale=log_scale).flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            return normalized_approx.reshape(n_grid_1d, n_grid_1d)
        else:
            return normalized_approx

    def compute_metrics(self, pred_type='pred', alphas=None):
        log_post_exact = self._exact_post_grid(shape_to_grid=False, log_scale=True)
        log_post_mean = self._mean_approx_grid(shape_to_grid=False, log_scale=True, pred_type=pred_type)
        log_post_eup = self._eup_approx_grid(shape_to_grid=False, log_scale=True, pred_type=pred_type)
        log_post_ep = self._ep_approx_grid(shape_to_grid=False, log_scale=True, pred_type=pred_type)

        mean_kl = kl_grid(log_post_exact, log_post_mean)
        eup_kl = kl_grid(log_post_exact, log_post_eup)
        ep_kl = kl_grid(log_post_exact, log_post_ep)
        
        alphas, mean_coverage, _ = coverage_curve(log_post_exact, log_post_mean, alphas=alphas)
        _, eup_coverage, _ = coverage_curve(log_post_exact, log_post_eup, alphas=alphas)
        _, ep_coverage, _ = coverage_curve(log_post_exact, log_post_ep, alphas=alphas)

        return {
            'kl': [mean_kl, eup_kl, ep_kl],
            'coverage': [mean_coverage, eup_coverage, ep_coverage],
            'alphas': alphas
        }

    def _get_plot_grid(self):
        grid_info = self.test_grid_info

        return grid_info["U1_grid"], grid_info["U2_grid"]

    def plot_exact_log_post(self):
        U1, U2 = self._get_plot_grid()
        log_post = self.test_grid_info["log_post_grid"]
        xlab, ylab = self.test_grid_info["axis_labels"]

        mappable, fig = plot_heatmap(U1, U2, log_post, title="Exact Posterior Log Density", 
                                     xlabel=xlab, ylabel=ylab)
        ax = fig.axes[0]
        ax.plot(*self.inv_prob.likelihood.par_true, "*", color="red", markersize=12)

        return fig, ax
    
    def plot_gp_bias(self, conditional=True, pred_type='pred', markersize=8, **kwargs):
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[pred_type].mean
        exact = self.test_grid_info["log_post"]
        biases = (means - exact).reshape(n,n)

        mappable, fig = plot_heatmap(U1, U2, biases, title="Emulator Bias", 
                                     xlabel=xlab, ylabel=ylab)
        ax = fig.axes[0]
        ax.plot(*self.inv_prob.likelihood.par_true, "*", color="red", markersize=12)
        ax.plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, ax
        

    def plot_gp_pred(self, conditional=True, pred_type='pred', markersize=8, **kwargs):
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[pred_type].mean.reshape(n,n)
        stdevs = jnp.sqrt(pred_dist[pred_type].variance).reshape(n,n)

        fig, axs, mappables = plot_independent_heatmaps(
            U1, U2,
            Z_list=[means, stdevs],
            titles=[f"{pred_type} mean", f"{pred_type} std dev"],
            xlab=xlab, ylab=ylab,
            **kwargs
        )

        # Add design points
        axs[0].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)
        axs[1].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs


    def plot_true_vs_gp_mean(self, conditional=True, latent_pred=False, markersize=8, **kwargs):
        dist_label = "latent" if latent_pred else "pred"
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[dist_label].mean.reshape(n,n)
        log_post_true = self.test_grid_info["log_post_grid"]

        fig, axs, mappables, cbar_obj = plot_shared_scale_heatmaps(
            U1, U2,
            Z_list=[log_post_true, means],
            titles=["exact log posterior", f"{dist_label} GP mean"],
            xlab=xlab, ylab=ylab,
            sharexy=True,
            **kwargs
        )

        # Add design points
        axs[0].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)
        axs[1].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs


    def plot_posterior_comparison(self, markersize=8, shared_scale=True, log_scale=False, 
                                  pred_type='pred', **kwargs):
        """ Plot exact vs plug-in mean vs EUP vs EP normalized densities """
        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]

        exact = self._exact_post_grid(log_scale=log_scale)
        mean = self._mean_approx_grid(log_scale=log_scale, pred_type=pred_type)
        eup = self._eup_approx_grid(log_scale=log_scale, pred_type=pred_type)
        ep = self._ep_approx_grid(log_scale=log_scale, pred_type=pred_type)

        param_list = [U1, U2]
        param_dict = {
            'Z_list': [exact, mean, eup, ep],
            'titles': ["exact", "mean", "eup", "ep"],
            'xlab': xlab,
            'ylab': ylab
        }

        if shared_scale:
            param_dict['sharexy'] = True
            fig, axs, mappables, cbar_obj = plot_shared_scale_heatmaps(*param_list, **param_dict, **kwargs)
        else:
            fig, axs, mappables = plot_independent_heatmaps(*param_list, **param_dict, **kwargs)

        # Add design points
        for i in range(4):
            axs[i].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs
    

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def average_pairwise_distance(x):
    """ x: (n, d) array. 
    Computes average Euclidean distance over set of pairwise distinct points
    """
    n = x.shape[0]
    diffs = x[:, None, :] - x[None, :, :]  # Shape: (n, n, d)
    dists = np.linalg.norm(diffs, axis=-1) # Shape: (n, n)

    # Distances between *distinct* pairs, so mask the diagonal
    mask = ~np.eye(n, dtype=bool)
    avg_dist = dists[mask].mean()
    return avg_dist


def average_pairwise_distance_per_dim(x):
    n, d = x.shape
    diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
    abs_diffs = np.abs(diffs)

    # Discard diagonal (i==j), only keep i != j pairs
    mask = ~np.eye(n, dtype=bool)
    abs_diffs_pairs = abs_diffs[mask].reshape(n * (n - 1), d)
    avg_dist_per_dim = abs_diffs_pairs.mean(axis=0)
    return avg_dist_per_dim


def plot_heatmap(X, Y, Z, title=None, ax=None,
                 cmap='viridis', shading='auto',
                 xlabel=None, ylabel=None,
                 cbar=True, cbar_kwargs=None):
    """
    Plot a single heatmap and return the mappable (QuadMesh/AxesImage).
    If ax is None, create a new figure+axis.

    Parameters
    ----------
    X, Y : 2D arrays
        Grid coordinates (as used by pcolormesh).
    Z : 2D array
        Values to plot; must match X/Y shape.
    title : str or None
    ax : matplotlib.axes.Axes or None
    cmap : str
    shading : str
    xlabel, ylabel : str or None
    cbar : bool
    cbar_kwargs : dict passed to fig.colorbar (optional)

    Returns
    -------
    mappable : the object returned by pcolormesh
    fig_or_none : If a new figure was created, returns that figure, otherwise None
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()

    # pcolormesh expects arrays shaped like the mesh
    m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if cbar:
        cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
        orientation = cbar_kwargs.pop('orientation', 'horizontal')
        fraction = cbar_kwargs.pop('fraction', 0.07)
        pad = cbar_kwargs.pop('pad', 0.15)
        fig_for_cb = created_fig if created_fig is not None else ax.figure
        cb = fig_for_cb.colorbar(m, ax=ax, orientation=orientation,
                                 fraction=fraction, pad=pad, **cbar_kwargs)
    else:
        cb = None

    return m, created_fig  # created_fig is None if ax was passed in


def plot_independent_heatmaps(X, Y, Z_list, titles=None,
                              xlab=None, ylab=None,
                              nrows=1, ncols=None, figsize=None,
                              cmap='viridis', shading='auto',
                              sharexy=False):
    """
    Plot multiple independent heatmaps (each with its own colorbar).
    Returns (fig, axs, mappables) and does NOT call plt.show().

    Parameters
    ----------
    X, Y : 2D arrays (same grid for all Zs)
    Z_list : list/iterable of 2D arrays (each same shape as X/Y)
    titles : list of str or None
    xlab, ylab : labels (applied to all subplots if provided)
    nrows : int
    ncols : int or None (auto computed if None)
    figsize : tuple or None (auto computed if None)
    cmap, shading : passed to pcolormesh
    sharexy : if True, share x/y axes between subplots (useful for aligned grids)

    Returns
    -------
    fig, axs_flat, mappables_list
    """
    nplots = len(Z_list)
    if ncols is None:
        ncols = int(ceil(nplots / nrows))

    # sensible default figure size if not provided:
    # give each subplot ~4 x 3 inches
    if figsize is None:
        per_ax_w, per_ax_h = 4.0, 3.0
        figsize = (per_ax_w * ncols, per_ax_h * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                            sharex=sharexy, sharey=sharexy)
    axs_flat = axs.reshape(-1)

    # normalize titles list length
    if titles is None:
        titles = [None] * nplots
    else:
        # extend or truncate to match nplots
        titles = list(titles) + [None] * max(0, nplots - len(titles))
        titles = titles[:nplots]

    mappables = []
    for i, Z in enumerate(Z_list):
        ax = axs_flat[i]
        m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap)
        mappables.append(m)

        if titles[i] is not None:
            ax.set_title(titles[i])
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)

        # add a horizontal colorbar under this axis with reasonable defaults
        cb = fig.colorbar(m, ax=ax, orientation='horizontal',
                          fraction=0.07, pad=0.18)

    # Hide any unused axes
    total_axes = nrows * ncols
    for j in range(nplots, total_axes):
        fig.delaxes(axs_flat[j])

    # Use tight_layout but leave space for colorbars; caller may further adjust
    fig.tight_layout()
    return fig, axs_flat[:nplots], mappables


def plot_shared_scale_heatmaps(
    X: np.ndarray,
    Y: np.ndarray,
    Z_list: Iterable[np.ndarray],
    titles: Optional[Iterable[Optional[str]]] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    nrows: int = 1,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    shading: str = 'auto',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar: bool = True,
    cbar_kwargs: Optional[dict] = None,
    sharexy: bool = False,
):
    """
    Plot multiple heatmaps that share a single global color scale (one colorbar).

    Parameters
    ----------
    X, Y : 2D arrays
        Grid coordinates (mesh) matching Z shapes (as used by pcolormesh).
    Z_list : iterable of 2D arrays
        Data arrays to plot; each must have the same shape as X/Y.
    titles : list/iterable of str or None
        Titles for each subplot (length will be truncated/extended to match the number of Zs).
    xlab, ylab : str or None
        Common x and y labels applied to all subplots if provided.
    nrows : int
        Number of subplot rows.
    ncols : int or None
        Number of subplot columns; auto-computed if None.
    figsize : (w, h) or None
        Figure size in inches; auto-computed if None (approx 4x3 inches per subplot).
    cmap : str
        Matplotlib colormap name.
    shading : str
        Passed to pcolormesh.
    vmin, vmax : float or None
        Global color limits. If None, computed from min/max of all Zs.
    cbar : bool
        Whether to draw the shared colorbar.
    cbar_kwargs : dict or None
        Extra kwargs passed to `fig.colorbar`. Common keys: orientation, fraction, pad, label, shrink, extend.
    sharexy : bool
        If True, share x and y axes between subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs_flat : array-like of Axes (length == nplots)
        Flat array of axes corresponding to each plot (in row-major order).
    mappables : list
        List of QuadMesh objects returned by pcolormesh for each subplot.
    cbar_obj : matplotlib.colorbar.Colorbar or None
        The shared colorbar object (None if cbar=False).
    """
    Z_list = list(Z_list)
    nplots = len(Z_list)
    if nplots == 0:
        raise ValueError("Z_list must contain at least one array to plot.")

    # compute ncols if necessary
    if ncols is None:
        ncols = int(ceil(nplots / nrows))

    # reasonable default figure size if not provided
    if figsize is None:
        per_ax_w, per_ax_h = 4.0, 3.0
        figsize = (per_ax_w * ncols, per_ax_h * nrows)

    # compute global vmin/vmax if not provided
    if vmin is None:
        vmin = min(np.nanmin(Z) for Z in Z_list)
    if vmax is None:
        vmax = max(np.nanmax(Z) for Z in Z_list)

    # prepare titles
    if titles is None:
        titles = [None] * nplots
    else:
        titles = list(titles) + [None] * max(0, nplots - len(list(titles)))
        titles = titles[:nplots]

    # create subplots
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False,
        sharex=sharexy, sharey=sharexy
    )
    axs_flat = axs.reshape(-1)

    mappables = []
    for i, Z in enumerate(Z_list):
        ax = axs_flat[i]
        m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
        mappables.append(m)

        if titles[i] is not None:
            ax.set_title(titles[i])
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)

    # hide unused axes
    total_axes = nrows * ncols
    for j in range(nplots, total_axes):
        fig.delaxes(axs_flat[j])

    # add a single shared colorbar (span all axes)
    cbar_obj = None
    if cbar:
        cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
        # defaults for a horizontal colorbar placed beneath the axes
        orientation = cbar_kwargs.pop('orientation', 'horizontal')
        fraction = cbar_kwargs.pop('fraction', 0.07)
        pad = cbar_kwargs.pop('pad', 0.18)
        # use the first mappable as the reference, and provide all axes for placement
        axes_for_cb = axs_flat[:nplots]  # only existing axes
        cbar_obj = fig.colorbar(
            mappables[0],
            ax=axes_for_cb,
            orientation=orientation,
            fraction=fraction,
            pad=pad,
            **cbar_kwargs
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18) # extra space for colorbar     
    # leave caller freedom to further adjust (e.g. fig.subplots_adjust(bottom=...))
    return fig, axs_flat[:nplots], mappables, cbar_obj


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp along `axis`. Returns array with `axis` eliminated."""
    a_max = np.max(a, axis=axis, keepdims=True)
    # If all entries are -inf, avoid NaN: keep a_max as -inf and sum_exp as 0
    sum_exp = np.sum(np.exp(a - a_max), axis=axis, keepdims=True)
    out = a_max + np.log(sum_exp)
    return np.squeeze(out, axis=axis)


def _normalize_over_grid(log_dens, weights=None, log_scale=True):
    """ Approximately normalize a density over a a grid of points

    `log_dens` represents a discretized unnormalized log density l = (l1, ..., lJ).
    This function computes log(l / Z) [log_scale=True] or l / Z [log_scale=False
    where Z = sum_{j} w_j exp{lj}, where Z typically represents a grid-based appriximation 
    of the normalizing constant. The `weights` argument provide the weights 
    (which need not be normalized).

    This function is vectorized to operate over the rows of `log_dens`, with each 
    row treated as a separate discretized density. 

    NOTE: assumes equally spaced grid, with equal grid spacing along both axes
    """

    if log_dens.ndim < 2:
        log_dens = log_dens.reshape(1,-1)
    if log_dens.ndim != 2:
        raise ValueError("`log_dens` must be a 2D array with shape (n, m).")
    n, m = log_dens.shape

    if weights is None:
        weights = np.ones(m, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (m,):
            raise ValueError(f"weights must have shape ({m},), got {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if np.all(weights == 0):
        raise ValueError("all weights are zero (no quadrature mass).")

    # log weights (if a weight is zero, log will be -inf, which is correct)
    logw = np.log(weights)

    # Compute logZ for each row of x:
    #   logZ_i = logsumexp(ell_i + logw)
    logdens_plus_logw = log_dens + logw[np.newaxis, :]   # (n, m)
    logZ = _logsumexp(logdens_plus_logw, axis=1)         # (n,)

    # Compute log normalized density: p_{ij} = ell_{ij} + log w_j - logZ_i
    log_dens_norm = logdens_plus_logw - logZ[:, np.newaxis] # (n, m)

    return log_dens_norm if log_scale else np.exp(log_dens_norm)


def estimate_ep_grid(
    logpi_samples: np.ndarray,
    weights: np.ndarray | None = None,
    return_se: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Estimate E_f[ pi(u_j; f) / Z(f) ] at grid nodes using Monte Carlo samples of log-densities.

    Args:
        logpi_samples: ndarray of shape (S, M). Row s contains [ell_j(f^{(s)})]_{j=1..M},
                       where ell_j = log pi(u_j; f^{(s)}).
                       S = number of Monte Carlo samples, M = number of grid nodes.
        weights: ndarray of shape (M,), quadrature weights w_j >= 0 for approximating the
                 integral Z(f) ≈ sum_j pi(u_j;f) * w_j. If None, equal weights are used.
                 Weights must be strictly non-negative and not all zero.
        return_se: whether to return Monte Carlo standard errors (True by default).

    Returns:
        mean_p: ndarray of shape (M,), the Monte Carlo estimate of E_f[ pi(u_j;f)/Z(f) ].
        log_mean_p: log of mean_p
        se_p: ndarray of shape (M,) giving Monte Carlo standard errors (if return_se True),
              otherwise None.

    Notes:
        - This function is numerically stable because it computes log Z(f) using log-sum-exp:
            log Z ≈ logsumexp( ell_j + log w_j ).
        - After subtracting log Z from ell_j + log w_j we exponentiate to obtain p_j(f) ∈ [0,1].
        - If memory is a concern (very large S), pass logpi_samples in batches and accumulate:
          accumulate sum_p and sum_p2; final mean = sum_p / S_total; var = (sum_p2/(S-1) - S_total/(S-1)*mean^2).
    """

    logp =  _normalize_over_grid(logpi_samples, weights=None, log_scale=True)
    S, M = logp.shape

    # Monte Carlo estimates: mean and (optionally) standard errors
    log_mean_p = _logsumexp(logp, axis=0) - np.log(S) # (M,)
    mean_p = np.exp(log_mean_p)                        
    se_p = None
    if return_se:
        if S > 1:
            # sample standard deviation / sqrt(S)
            std_p = np.std(np.exp(logp), axis=0, ddof=1) # TODO: improve numerical stability here
            se_p = std_p / np.sqrt(S)
        else:
            se_p = np.full(M, np.nan)

    return mean_p, log_mean_p, se_p


def kl_grid(logp, logq):
    """
    KL(p || q) = \int p * (log p - log q) dx.
    Numerically stable: if q has zeros where p>0, KL is large/infinite; we floor q.
    Returns KL value (scalar).

    Assumes equally spaced grid in both axes.
    """
    p = np.exp(logp)
    integrand = p * (logp - logq)
    kl = integrand.mean()
    return kl


def hpd_region_mask_from_logp(logp, alpha, dx, dy):
    """
    Given logp (normalized log density) on grid, produce boolean mask for HPD region
    that contains mass alpha (i.e., highest logp until cumulative mass >= alpha).
    """
    nx, ny = logp.shape
    flat_idx = np.argsort(logp.ravel())[::-1]  # indices sorted descending by logp
    # To accumulate mass, need p (from logp) and dx*dy later
    return flat_idx  # return ordering for later usage


def coverage_curve(logp_true, logp_approx, alphas=None):
    """
    Compute coverage curve: for each alpha in alphas, construct approximate HPD region
    (based on approx logp), and compute mass under true p contained in that region.

    Returns:
        alphas: array
        coverage: array same length - true mass inside approx HPD(alpha)
        calibration_error: mean absolute deviation between coverage and nominal alpha
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.99, 99)

    logp_t = logp_true
    logp_a = logp_approx
    p_t = np.exp(logp_t)
    p_a = np.exp(logp_a)

    order = np.argsort(logp_a)[::-1]    # descending indices
    approx_cum = np.cumsum(p_a[order])  # sums to 1

    # For each alpha, find minimal index k where approx_cum[k] >= alpha
    coverage = []
    for alpha in alphas:
        k = np.searchsorted(approx_cum, alpha, side='left')
        idxs = order[:k+1] if k < order.size else order # indices in the HPD region
        true_mass = p_t[idxs].sum()
        coverage.append(true_mass)

    coverage = np.array(coverage)
    calib_error = np.mean(np.abs(coverage - alphas))
    return alphas, coverage, calib_error
