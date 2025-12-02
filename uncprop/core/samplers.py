# uncprop/core/samplers.py


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