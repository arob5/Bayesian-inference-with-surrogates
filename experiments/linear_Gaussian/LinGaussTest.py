# experiments/test/linear_Gaussian/LinGaussTest.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_triangular

from Gaussian import Gaussian
from helper import get_col_hist_grid, get_trace_plots, get_random_corr_mat

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import MarkovKernel, GaussMetropolisKernel, DiscretePCNKernel, UncalibratedDiscretePCNKernel, mvn_logpdf


class LinGaussTest:
    def __init__(self, rng, d, n, Sig_scale=1.0, Q_scale=1.0, C0_scale=1.0):
        Sig = Sig_scale**2 * get_random_corr_mat(n, rng)
        Q = Q_scale**2 * get_random_corr_mat(n, rng)
        C0 = C0_scale**2 * get_random_corr_mat(d, rng)
        m0 = rng.normal(size=d)

        # Exact inverse problem.
        self.rng = rng
        self.d = d
        self.n = n
        self.u_names = [f"u{i+1}" for i in range(d)]
        self.G = rng.normal(size=(n,d)) # linear forward model.
        self.prior = Gaussian(mean=m0, cov=C0, rng=rng)
        self.noise = Gaussian(cov=Sig, rng=rng, store="both")
        self.u_true = self.prior.sample()
        self.y = self.G @ self.u_true + self.noise.sample()
        self.post = self.prior.invert_affine_Gaussian(self.y, A=self.G,
                                                      cov_noise=self.noise.cov)

        # Well-calibrated surrogate (sampling bias mean r from N(0,Q)).
        r = Gaussian(cov=Q, rng=rng).sample()
        self.e = Gaussian(mean=r, cov=Q, rng=rng) # Random bias; emulator is Gu + e.

        # Surrogate-based inversion.
        self.ep_post = self.get_ep_rv()
        self.eup_post = self.get_eup_rv()

        # MCMC samplers.
        self.samplers = {"mwg-eup": self.get_mwg_eup_sampler(),
                         "rk": self.get_rk_sampler(),
                         "rk-pcn": self.get_rk_pcn_sampler()}

    def get_ep_rv(self):
        L_Sig = self.noise.chol
        C1 = solve_triangular(L_Sig, self.G, lower=True)
        C2 = solve_triangular(L_Sig.T, C1, lower=False)
        B = -self.post.cov @ C2.T
        ep = self.e.convolve_with_Gaussian(A=B, b=self.post.mean, cov_new=self.post.cov)
        return ep

    def get_eup_rv(self):
        eup = self.prior.invert_affine_Gaussian(self.y, A=self.G, b=self.e.mean,
                                                cov_noise=self.noise.cov + self.e.cov)
        return eup

    def run_sampler(self, sampler_name, n_samp, sampler_kwargs=None, plot_kwargs=None):
        """
        Runs MCMC sampler, collects samples, and drops burn-in. Returns
        `n_samp` samples after burn-in.
        """
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        sampler = self.samplers[sampler_name]
        sampler.sample(num_steps=2*n_samp, **sampler_kwargs)

        # Store samples in array.
        len_trace = len(sampler.trace)
        itr_range = np.arange(len_trace-n_samp, len_trace)
        samp = np.empty((n_samp, self.d))

        for samp_idx,trace_idx in enumerate(itr_range):
            samp[samp_idx,:] = sampler.trace[trace_idx].primary["u"]

        return samp, self.get_trace_plot(samp, **plot_kwargs)

    def reset_samplers(self):
        for sampler in self.samplers.values():
            sampler.reset()

    def get_sample_list(self, n_samp, include=None):
        # `include` can include ["post", "eup", "ep", "mwg-eup", "rk", "rk-pcn"]
        samp_list = []
        labels = []
        if include is None:
            include = ["post", "eup", "ep"]

        if "post" in include:
            samp_list.append(self.post.sample(n_samp))
            labels.append("post")
        if "ep" in include:
            samp_list.append(self.ep_post.sample(n_samp))
            labels.append("ep")
        if "eup" in include:
            samp_list.append(self.eup_post.sample(n_samp))
            labels.append("eup")
        if "mwg-eup" in include:
            samp_list.append(self.run_sampler("mwg-eup", n_samp)[0])
            labels.append("mwg-eup")
        if "rk" in include:
            samp_list.append(self.run_sampler("rk", n_samp)[0])
            labels.append("rk")
        if "rk-pcn" in include:
            samp_list.append(self.run_sampler("rk-pcn", n_samp)[0])
            labels.append("rk-pcn")

        return samp_list, labels

    def get_hist_plot(self, n_samp=100000, include=None, plot_kwargs=None):
        samp_list, plot_labs = self.get_sample_list(n_samp, include)
        if plot_kwargs is None:
            plot_kwargs = {}

        fig = get_col_hist_grid(*samp_list, plot_labs=plot_labs, col_labs=self.u_names,
                                density=True, **plot_kwargs)
        return fig

    def get_trace_plot(self, samp, plot_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = {}
        fig = get_trace_plots(samp, col_labs=self.u_names, **plot_kwargs)
        return fig

    def get_mwg_eup_sampler(self, u_prop_scale=0.1, pcn_cor=0.99):
        """
        Exactly targets the EUP.
        """
        L_noise = self.noise.chol

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

        # Target density.
        def ldens_post(state):
            fwd = self.G @ state.primary["u"] + state.primary["e"]
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

        target = TargetDensity(LogDensityTerm("post", ldens_post))

        # u and e updates.
        ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                      term_subset="post", block_vars="u", rng=self.rng)
        ker_e = DiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                  cor_param=pcn_cor, term_subset="post",
                                  block_vars="e", rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state,
                               kernels=[ker_u, ker_e], rng=self.rng)
        return alg

    def get_rk_sampler(self, u_prop_scale=0.1):
        L_noise = self.noise.chol

        # Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample()})

        # Noisy target density.
        def ldens_post_noisy(state):
            fwd = self.G @ state.primary["u"] + self.e.sample()
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

        target = TargetDensity(LogDensityTerm("post", ldens_post_noisy), use_cache=False)

        # Metropolis-Hastings updates.
        ker = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov, rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state, kernels=ker, rng=self.rng)
        return alg

    def get_rk_pcn_sampler(self, u_prop_scale=0.1, pcn_cor=0.9):
        L_noise = self.noise.chol

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

        # Target density.
        def ldens_post(state):
            fwd = self.G @ state.primary["u"] + state.primary["e"]
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])
        target = TargetDensity(LogDensityTerm("post", ldens_post))

        # u and e updates.
        ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                      term_subset="post", block_vars="u", rng=self.rng)
        ker_e = UncalibratedDiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                              cor_param=pcn_cor, block_vars="e", rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e], rng=self.rng)
        return alg

    def direct_sample_ep(self, n_samp=100000):
        """
        An alternative method to sample the EP. The analytical approach is
        faster, this is mostly just used for validation.
        """
        samp = np.empty((n_samp, self.d))
        for i in range(n_samp):
            samp[i,:] = self.prior.invert_affine_Gaussian(y, A=self.G, b=self.e.sample(),
                                                          cov_noise=self.noise.cov).sample()
        return samp

    def calc_coverage(self, probs=None):
        """
        Computes marginal (one dim at a time) coverage for EP and EUP, with
        respect to the baseline exact posterior. Returns dictionary with
        keys "ep", "eup", and "probs". "probs" contains the list of
        coverage levels (e.g., 0.9 = 90% coverage interval). The other two
        elements are each `(m,d)` arrays, where `m = len(probs)`.
        The `(i,j)` element of the array is the actual coverage of the `jth`
        marginal of EUP or EP (with respect to the exact posterior baseline)
        at level `probs[i]`.
        """

        # Lower and upper quantiles defining coverage sets.
        if probs is None:
            probs = np.arange(0.1, 1.01, step=0.1)
        lower = 0.5 * (1 - probs)
        upper = 0.5 * (1 + probs)
        intervals = np.array([lower, upper])

        m = len(probs)
        actual_coverage = {"ep": np.empty((m, self.d)),
                           "eup": np.empty((m, self.d)),
                           "probs": probs}

        # Loop over 1d marginals.
        for j in range(self.d):
            # Nominal coverage.
            nominal_ep = norm.ppf(intervals, loc=self.ep_post.mean[j],
                                  scale=np.sqrt(self.ep_post.cov[j,j]))
            nominal_eup = norm.ppf(intervals, loc=self.eup_post.mean[j],
                                   scale=np.sqrt(self.eup_post.cov[j,j]))

            # Actual coverage.
            actual_ep = norm.cdf(nominal_ep, loc=self.post.mean[j],
                                 scale=np.sqrt(self.post.cov[j,j]))
            actual_eup = norm.cdf(nominal_eup, loc=self.post.mean[j],
                                  scale=np.sqrt(self.post.cov[j,j]))
            actual_coverage["ep"][:,j] = actual_ep[1,:] - actual_ep[0,:]
            actual_coverage["eup"][:,j] = actual_eup[1,:] - actual_eup[0,:]

        return actual_coverage

    def plot_coverage(self, coverage_info=None, probs=None, nrows=1,
                      ncols=None, figsize=(5,4)):

        if coverage_info is None:
            coverage_info = self.calc_coverage(probs)
        if ncols is None:
            ncols = int(np.ceil(self.d / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(self.d):
            ax = axs[j]
            ax.plot(coverage_info["probs"], coverage_info["ep"][:,j], label="ep")
            ax.plot(coverage_info["probs"], coverage_info["eup"][:,j], label="eup")
            ax.set_title(self.u_names[j])
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Actual Coverage")

            # Add line y = x
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            y = x
            ax.plot(x, y, color="red", linestyle="--")
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(self.d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig
