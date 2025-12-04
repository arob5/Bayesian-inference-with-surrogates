# uncprop/core/samplers.py

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Callable

import jax
import blackjax
import jax.random as jr
import jax.numpy as jnp
from blackjax.base import (
    UpdateFn,
    Position,
    State,
    Info,
)

from uncprop.core.inverse_problem import (
    PRNGKey, 
    Array,
    ArrayLike,
)


def mcmc_loop(key: PRNGKey, 
              kernel: UpdateFn, 
              initial_state: State, 
              num_samples: int = 4000):
    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jax.random.split(key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def init_nuts_kernel(key: PRNGKey, 
                     logdensity: Callable,
                     initial_position: Position,
                     num_warmup_steps: int = 1000):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    warmup_key, _ = jax.random.split(key, 2)
    (initial_state, nuts_params), _ = warmup.run(warmup_key, initial_position, num_steps=1000)
    kernel = blackjax.nuts(logdensity, **nuts_params).step

    return initial_state, kernel


def stack_dict_arrays(data_dict: dict[str, ArrayLike], 
                      names: list[str] | None = None) -> Array:
    """
    Given a dictionary of arrays, each of shape (n,) or (n,d) (with d allowed to vary
    by array), concatenate to array of shape (n, d_total). The arrays are concatenated
    left-to-right in the order specified by `names`.
    """
    if names is None:
        names = list(data_dict.keys())

    arrays = [jnp.asarray(data_dict[name]) for name in names]
    arrays = jax.tree.map(tree=arrays, f=_ensure_2d)

    n_rows = arrays[0].shape[0]
    assert all(a.shape[0] == n_rows for a in arrays), "All arrays must have same number of rows"
    return jnp.concatenate(arrays, axis=1) 


def get_trace_plots(trace: dict[str, Array], 
                    names: list[str] | None = None,
                    num_rows: int = 1,
                    figsize: tuple = (15, 6)):

    if names is None:
        names = list(trace.keys())

    num_plots = len(names)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_plots-num_rows+1)
    if num_plots > 1:
        axs = axs.ravel()

    for i in range(num_plots):
        ax = axs[i]
        nm = names[i]
        trajectory = trace[nm]

        ax.plot(trajectory)
        ax.set_xlabel('samples')
        ax.set_ylabel(nm)
    
    return fig, axs


def _ensure_2d(x):
    x = jnp.asarray(x)
    assert x.ndim <= 2

    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x



if __name__ == '__main__':
    import jax.scipy.stats as stats
    from datetime import date

    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    loc, scale = 10, 20
    observed = np.random.normal(loc, scale, size=1_000)

    def logdensity_fn(loc, log_scale, observed=observed):
        """Univariate Normal"""
        scale = jnp.exp(log_scale)
        logjac = log_scale
        logpdf = stats.norm.logpdf(observed, loc, scale)
        return logjac + jnp.sum(logpdf)
    logdensity = lambda x: logdensity_fn(**x)

    initial_position = {"loc": 1.0, "log_scale": 1.0}

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)

    kernel = blackjax.nuts(logdensity, **parameters).step
    
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, 1_000)

    mcmc_samples = states.position
    mcmc_samples["scale"] = jnp.exp(mcmc_samples["log_scale"]).block_until_ready()

    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
    ax.plot(mcmc_samples["loc"])
    ax.set_xlabel("Samples")
    ax.set_ylabel("loc")

    ax1.plot(mcmc_samples["scale"])
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("scale")
    fig.savefig('/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper/uncprop/test.png')



# def _set_default_sampler(self, proposal_cov: Array | None = None, **kwargs):
#         """ Defaults to Metropolis-Hastings. """
        
#         if proposal_cov is None:
#             proposal_cov = np.identity(self.dim)

#         # Extended state space. Initialize state via prior sample.
#         state = State(primary={"u": self.prior.sample()})

#         # Target density.
#         post_log_dens = lambda state: self.log_posterior_density(state.primary["u"])
#         target = TargetDensity(LogDensityTerm("post", post_log_dens))

#         # Metropolis-Hastings kernel with Gaussian proposal.
#         mh_kernel = GaussMetropolisKernel(target, proposal_cov=proposal_cov, rng=self.rng)

#         # Sampler
#         sampler = BlockMCMCSampler(target, initial_state=state, kernels=mh_kernel, rng=self.rng)
#         return sampler
    

#     def sample_posterior(self, n_step: int, burn_in_start: int | None = None, 
#                          sampler_kwargs=None, plot_kwargs=None):
#         """
#         Runs MCMC sampler, collects samples, and drops burn-in. Returns
#         `n_samp` samples after burn-in. Default burn-in is to take second
#         half of samples.
#         """
#         if sampler_kwargs is None:
#             sampler_kwargs = {}
#         if plot_kwargs is None:
#             plot_kwargs = {}

#         self.sampler.sample(num_steps=n_step, **sampler_kwargs)

#         # Store samples in array.
#         burn_in_start = burn_in_start or round(n_step / 2)
#         itr_range = np.arange(burn_in_start, len(self.sampler.trace))
#         n_samp = len(itr_range)
#         samp = np.empty((n_samp, self.dim))

#         for samp_idx, trace_idx in enumerate(itr_range):
#             samp[samp_idx,:] = self.sampler.trace[trace_idx].primary["u"]

#         return samp, self.get_trace_plot(samp, **plot_kwargs)

#     def reset_sampler(self):
#         self.sampler.reset()

#     def get_trace_plot(self, samp, nrows=1, ncols=None, col_labs=None, figsize=(5,4), plot_kwargs=None):
#         n_itr, n_cols = samp.shape
#         x = np.arange(n_itr)

#         if plot_kwargs is None:
#             plot_kwargs = {}

#         if ncols is None:
#             ncols = int(np.ceil(n_cols / nrows))

#         if col_labs is None:
#             col_labs = self.par_names

#         fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
#         axs = np.array(axs).reshape(-1)
#         for col in range(n_cols):
#             ax = axs[col]
#             ax.plot(x, samp[:,col], **plot_kwargs)
#             ax.set_title(col_labs[col])
#             ax.set_xlabel("Iteration")
#             ax.set_ylabel("Value")

#         # Hide unused axes and close figure.
#         for k in range(n_cols, nrows*ncols):
#             fig.delaxes(axs[k])
#         plt.close(fig)

#         return fig