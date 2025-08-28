# experiments/test/linear_Gaussian/LinGaussTest.py

import numpy as np
from scipy.linalg import solve_triangular

from Gaussian import Gaussian
from helper import get_col_hist_grid, get_random_corr_mat

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

        return samp_list, labels

    def get_hist_plot(self, n_samp=100000, include=None, plot_kwargs=None):
        samp_list, plot_labs = self.get_sample_list(n_samp, include)
        if plot_kwargs is None:
            plot_kwargs = {}

        fig = get_col_hist_grid(*samp_list, plot_labs=plot_labs, col_labs=self.u_names,
                                density=True, **plot_kwargs)
        return fig
