# uncprop/models/linear_Gaussian/LinGaussInvProb.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_triangular

from uncprop.models.linear_Gaussian.Gaussian import Gaussian


class LinGaussInvProb:
    # Exact (no surrogate) linear Gaussian inverse problem.

    def __init__(self, rng, G, m0=None, C0=None, Sig=None, y=None):
        self.rng = rng
        self.n = G.shape[0]
        self.d = G.shape[1]
        self.u_names = [f"u{i+1}" for i in range(self.d)]

        # Linear forward model
        self.G = G

        # Prior on parameter
        if m0 is None:
            m0 = np.zeros(self.d)
        if C0 is None:
            C0 = np.identity(self.d)
        self.prior = Gaussian(mean=m0, cov=C0, rng=rng)

        # Observation noise
        if Sig is None:
            Sig = np.identity(self.n)
        self.noise = Gaussian(cov=Sig, rng=rng, store="both")

        # Ground truth parameter and observed data
        if y is None:
            self.u_true = self.prior.sample()
            self.noise_realization = self.noise.sample()
            self.g_true = self.G @ self.u_true
            self.y = self.g_true + self.noise_realization
        else:
            self.y = y
            self.u_true = None
            self.noise_realization = None
            self.g_true = None

        # Exact posterior
        self.post = self.prior.invert_affine_Gaussian(self.y, A=self.G,
                                                      cov_noise=self.noise.cov)


    def plot_marginals(self, nrows=1, ncols=None, figsize=(5,4)):
        # Plot marginal Gaussian prior and posterior densities.

        if ncols is None:
            ncols = int(np.ceil(self.d / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(self.d):
            ax = axs[j]
            m0,s0 = self.prior.mean[j], np.sqrt(self.prior.cov[j,j])
            m,s = self.post.mean[j], np.sqrt(self.post.cov[j,j])

            bounds = norm.ppf([0.01, 0.99], loc=m0, scale=s0)
            x = np.linspace(bounds[0], bounds[1], 100)
            ax.plot(x, norm.pdf(x, loc=m0, scale=s0), label="prior")
            ax.plot(x, norm.pdf(x, loc=m, scale=s), label="posterior")
            ax.set_title(self.u_names[j])
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(self.d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig

    def plot_G_singular_vals(self, semilog=False, whiten=False):
        forward_model = self.G
        if whiten:
            forward_model = solve_triangular(self.noise.chol, forward_model, lower=True)
            title = "singular values of whitened G"
        else:
            title = "singular values of G"

        U, s, Vh = np.linalg.svd(forward_model)

        fig, ax = plt.subplots(1,1)

        if semilog:
            ax.semilogy(s)
        else:
            ax.plot(np.arange(len(s)), s)

        ax.set_title(title)
        ax.set_xlabel("index")
        ax.set_ylabel("singular value")

        plt.close(fig)
        return fig
