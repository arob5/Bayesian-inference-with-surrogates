# uncprob/models/vsem/inverse_problem.py
"""
Defines the Bayesian inverse problem for the VSEM experiment.
"""
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Protocol
from numpyro.distributions import Uniform
from scipy.stats import qmc

from uncprop.models.vsem import vsem_jax as vsem
from uncprop.core.inverse_problem import (
    Prior,
    LogLikelihood,
    Posterior,
    Array,
    PRNGKey,
)

import sys
sys.path.append("./../linear_Gaussian/")
from Gaussian import Gaussian


# def sample_lhc(self, n=1):
#     """ Latin hypercube sampling """
#     lhc = qmc.LatinHypercube(d=self.dim)

#     # Samples unit hypercube [0, 1)^d
#     samp = lhc.random(n=n)

#     # Scale based on prior bounds
#     l, u = self.support_bounds
#     samp = qmc.scale(samp, l, u)

#     return samp


class VSEMPrior(Prior):
    """
    The VSEM prior primarily represents the prior distribution over the VSEM 
    parameters that will be calibrated in the experiment. However, it also stores
    `_dists`, which defines the prior over all VSEM parameters. Additional methods
    beyond the standard Prior interface provide access to this extended prior.
    """

    _dists = {
        "kext": Uniform(0.4, 1.0),
        "lar" : Uniform(0.2, 3.0),
        "lue" : Uniform(5e-04, 4e-03),
        "gamma": Uniform(2e-01, 6e-01),
        "tauv": Uniform(5e+02, 3e+03),
        "taus": Uniform(4e+03, 5e+04),
        "taur": Uniform(5e+02, 3e+03), 
        "av": Uniform(0.4, 1.0),
        "veg_init": Uniform(0.0, 10.0), 
        "soil_init": Uniform(0.0, 30.0),
        "root_init": Uniform(0.0, 10.0)
    }

    def __init__(self, par_names: list[str] | None = None):
        """ par_names defines the set of calibration parameters. """
        self._par_names = par_names or vsem.get_vsem_par_names()
        low = jnp.array([dist.low for dist in self.dists])
        high = jnp.array([dist.high for dist in self.dists])
        self._prior_rv = Uniform(low, high)

    @property
    def dists(self):
        return {par: self._dists[par] for par in self._par_names}

    @property
    def dim(self):
        return len(self._par_names)
    
    @property
    def par_names(self):
        return self._par_names
    
    @property
    def support_bounds(self):
        return self._prior_rv.low, self._prior_rv.high
    
    def sample(self, key: PRNGKey, n: int = 1):
        return self._prior_rv.sample(key, sample_shape=(n,))

    def log_density(self, x):
        # TODO: Uniform doesnt throw error or return -Inf if value is outside of support.
        #       should maybe manually check for this case.
        
        l = self._prior_rv.log_prob(x)

        # sum over batch axis
        return jnp.sum(l, axis=tuple(range(-1, 0)))
    
    def sample_all_vsem_params(self, key: PRNGKey):
        """
        Return dictionary representing a single sample from all of the parameters, 
        not just the calibration parameters.
        """
        samp = {}
        for par_name, prior in self._dists.items():
            samp[par_name] = prior.sample(key)

        return samp


class VSEMLikelihood:
    """
    Defined to conform to the LogDensity protocol. 
    """

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