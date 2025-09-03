# experiments/linear_Gaussian/run_coverage_test.py
import os

import numpy as np
import matplotlib.pyplot as plt

from Gaussian import Gaussian
from LinGaussTest import LinGaussInvProb, LinGaussTest
from helper import get_random_corr_mat

def run_coverage_test(rng, n_reps, d, n, Q_scale=1.0,
                      C0_scale=1.0, Sig_scale=1.0):

    # Exact inverse problem (fixed throughout test)
    inv_prob = LinGaussInvProb(rng, d, n, C0_scale, Sig_scale)
    plt_inv_prob = inv_prob.plot_marginals()

    # Surrogate covariance
    Q = Q_scale**2 * get_random_corr_mat(n, rng)

    # Quantiles to compute for coverage metrics
    probs = np.arange(0.1, 1.01, step=0.1)
    n_probs = len(probs)
    ep_coverage = np.empty((n_reps, n_probs, inv_prob.d))
    eup_coverage = np.empty((n_reps, n_probs, inv_prob.d))

    for i in range(n_reps):
        test = LinGaussTest(inv_prob, Q, r=None) # Samples r from N(0,Q)
        res = test.calc_coverage(probs=probs)
        ep_coverage[i,:,:] = res["ep"]
        eup_coverage[i,:,:] = res["eup"]

    return inv_prob, ep_coverage, eup_coverage, probs


def plot_coverage_reps(ep_coverage, eup_coverage, probs, q_min=0.05,
                       q_max=0.95, nrows=1, ncols=None, figsize=(5,4)):

        d = ep_coverage.shape[2]
        if ncols is None:
            ncols = int(np.ceil(d / nrows))

        ep_m = np.median(ep_coverage, axis=0)
        eup_m = np.median(eup_coverage, axis=0)
        ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)
        eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(d):
            ax = axs[j]

            ax.fill_between(probs, ep_q[0,:,j], ep_q[1,:,j], color='blue', alpha=0.3, label='ep')
            ax.fill_between(probs, eup_q[0,:,j], eup_q[1,:,j], color='red', alpha=0.3, label='eup')
            ax.plot(probs, ep_m[:,j], color='blue', label='ep')
            ax.plot(probs, eup_m[:,j], color='red', label='eup')
            ax.set_title(j)
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Actual Coverage")

            # Add line y = x
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            y = x
            ax.plot(x, y, color="red", linestyle="--")
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig


if __name__ == "__main__":
    rng = np.random.default_rng(532124)
    d = 3  # Parameter dimension
    n = 20 # Observation dimension
    n_reps = 100

    out_dir = "/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper/" \
              "experiments/linear_Gaussian/out/coverage"

    inv_prob, ep_coverage, eup_coverage, probs = run_coverage_test(rng, n_reps, d, n,
                                                         Q_scale=1.0, C0_scale=1.0,
                                                         Sig_scale=1.0)

    ep_coverage.save(os.path.join(out_dir, "ep_data.npy"))
    eup_coverage.save(os.path.join(out_dir, "eup_data.npy"))
