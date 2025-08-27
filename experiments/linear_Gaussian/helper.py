import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

from Gaussian import Gaussian
from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import GaussMetropolisKernel, DiscretePCNKernel, mvn_logpdf

def get_col_hist_grid(*arrays, bins=30, nrows=1, ncols=None, figsize=(5,4),
                      col_labs=None, plot_labs=None, density=True, hist_kwargs=None):
    """
    Given multiple (n, d) arrays (same d), return one matplotlib Figure per column.
    Each Figure contains one histogram per array (for that column).

    Parameters:
        *arrays: arbitrary number of numpy arrays, each shape (n, d)
        bins: number of histogram bins (default=30)
        hist_kwargs: dict of extra kwargs for plt.hist (optional)
        col_labs: list of d names associated with the columns.
        plot_labs: list of len(arrays) names associated with the arrays.

    Returns:
        figs: list of matplotlib Figure objects (len=d)
    """

    n_cols = arrays[0].shape[1]
    if not all(a.shape[1] == n_cols for a in arrays):
        raise ValueError("All arrays must have the same number of columns")

    if hist_kwargs is None:
        hist_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_cols / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_cols)]
    if plot_labs is None:
        plot_labs = [f"Array {i+1}" for i in range(len(arrays))]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_cols):
        ax = axs[col]
        for idx, arr in enumerate(arrays):
            ax.hist(arr[:, col], bins=bins, alpha=0.5, label=plot_labs[idx],
                    density=density, **hist_kwargs)
        ax.set_title(col_labs[col])
        ax.legend()

    # Hide unused axes
    for k in range(n_cols, nrows*ncols):
        fig.delaxes(axs[k])
    return fig


def plot_trace(samp_arr, nrows=1, ncols=None, figsize=(5,4),
               col_labs=None, plot_kwargs=None):
    """
    Generate one trace plot per column of `samp_arr`.

    Parameters:
        arr: numpy array of shape (n, d)
        col_labs: list of labels associated with each column of `samp_arr`.

    Returns:
        fig: matplotlib Figure object
        ax: matplotlib Axes object
    """
    n_itr, n_cols = samp_arr.shape
    x = np.arange(n_itr)

    if plot_kwargs is None:
        plot_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_cols / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_cols)]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_cols):
        ax = axs[col]
        ax.plot(x, samp_arr[:,col], **plot_kwargs)
        ax.set_title(col_labs[col])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    # Hide unused axes
    for k in range(n_cols, nrows*ncols):
        fig.delaxes(axs[k])

    return fig

# ------------------------------------------------------------------------------
# Approximate MCMC schemes.
# ------------------------------------------------------------------------------

def get_mwg_sampler(y, u, G, noise, e, rng, u_prop_scale=0.1, pcn_cor=0.9):
    L_noise = noise.chol

    # Extended state space. Initialize state via prior sample.
    state = State(primary={"u": u.sample(), "e": e.sample()})

    # Target density.
    # ldens_surrogate = lambda state: e.log_p(state.primary["e"])
    def ldens_post(state):
        fwd = G @ state.primary["u"] + state.primary["e"]
        return mvn_logpdf(y, mean=fwd, L=L_noise) + u.log_p(state.primary["u"])

    target = TargetDensity(LogDensityTerm("post", ldens_post))

    # target = TargetDensity([LogDensityTerm("surrogate", ldens_surrogate),
    #                         LogDensityTerm("post", ldens_post)])

    # u and e updates.
    ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*u.cov,
                                  term_subset="post", block_vars="u", rng=rng)
    ker_e = DiscretePCNKernel(target, mean_Gauss=e.mean, cov_Gauss=e.cov,
                              cor_param=pcn_cor, term_subset="post",
                              block_vars="e", rng=rng)

    # Sampler
    mwg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e])
    return mwg

def get_naive_cut_sampler(y, u, G, noise, e, rng, u_prop_scale=0.1):
    L_noise = noise.chol

    # Initialize state via prior sample.
    state = State(primary={"u": u.sample()})

    # Noisy target density.
    def ldens_post_noisy(state):
        fwd = G @ state.primary["u"] + e.sample()
        return mvn_logpdf(y, mean=fwd, L=L_noise) + u.log_p(state.primary["u"])

    target = TargetDensity(LogDensityTerm("post", ldens_post_noisy), use_cache=False)

    # Metropolis-Hastings updates.
    ker = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*u.cov, rng=rng)

    # Sampler
    cut_alg = BlockMCMCSampler(target, initial_state=state, kernels=ker)
    return cut_alg

# ------------------------------------------------------------------------------
# Exact computation for EP and EUP
# ------------------------------------------------------------------------------

def get_ep_rv(y, u, G, noise, e):
    u_post = u.invert_affine_Gaussian(y, A=G, cov_noise=noise.cov, store="both")
    L_Sig = noise.chol
    C1 = solve_triangular(L_Sig, G, lower=True)
    C2 = solve_triangular(L_Sig.T, C1, lower=False)
    B = -u_post.cov @ C2.T
    u_ep = e.convolve_with_Gaussian(A=B, b=u_post.mean, cov_new=u_post.cov)
    return u_ep

def get_eup_rv(y, u, G, noise, e):
    u_eup = u.invert_affine_Gaussian(y, A=G, b=e.mean, cov_noise=noise.cov+e.cov)
    return u_eup

def direct_sample_ep(y, u, G, noise, e, num_samp):
    samp = np.empty((num_samp, u.dim))
    for i in range(num_samp):
        samp[i,:] = u.invert_affine_Gaussian(y, A=G, b=e.sample(), cov_noise=noise.cov).sample()
    return samp
