# experiments/test/linear_Gaussian/LinGaussTest.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Protocol

import vsem

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import (
    MarkovKernel, 
    GaussMetropolisKernel, 
    DiscretePCNKernel, 
    UncalibratedDiscretePCNKernel, 
    mvn_logpdf
)


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


class InvProb:

    def __init__(self, rng, prior: Prior, likelihood: Likelihood, sampler=None, **sampler_kwargs):
        self.rng = rng
        self.prior = prior
        self.likelihood = likelihood
        self.dim = int(self.prior.dim)
        self.par_names = ["u"+str(i) for i in range(1, self.dim+1)]
        self.sampler = self._set_default_sampler(**sampler_kwargs) if sampler is None else sampler
        self.posterior = None 


    def _set_default_sampler(self, proposal_cov: Array | None = None, **kwargs):
        """ Defaults to Metropolis-Hastings. """
        
        if proposal_cov is None:
            proposal_cov = np.identity(self.dim)

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample()})

        # Target density.
        post_log_dens = lambda state: self.likelihood.log_density(state.primary["u"])
        target = TargetDensity(LogDensityTerm("post", post_log_dens))

        # Metropolis-Hastings kernel with Gaussian proposal.
        mh_kernel = GaussMetropolisKernel(target, proposal_cov=proposal_cov, rng=self.rng)

        # Sampler
        sampler = BlockMCMCSampler(target, initial_state=state, kernels=mh_kernel, rng=self.rng)
        return sampler
    

    def sample_posterior(self, n_samp: int, sampler_kwargs=None, plot_kwargs=None):
        """
        Runs MCMC sampler, collects samples, and drops burn-in. Returns
        `n_samp` samples after burn-in.
        """
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        self.sampler.sample(num_steps=2*n_samp, **sampler_kwargs)

        # Store samples in array.
        len_trace = len(self.sampler.trace)
        itr_range = np.arange(len_trace-n_samp, len_trace)
        samp = np.empty((n_samp, self.dim))

        for samp_idx, trace_idx in enumerate(itr_range):
            samp[samp_idx,:] = self.sampler.trace[trace_idx].primary["u"]

        return samp, self.get_trace_plot(samp, **plot_kwargs)

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
    """ Uncertainty propagation experiment for surrogate modeling for VSEM inverse problem. """
    pass

