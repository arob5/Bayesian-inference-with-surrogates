# experiments/linear_Gaussian/run_coverage_test.py
import os

import numpy as np
import matplotlib.pyplot as plt

from Gaussian import Gaussian
from LinGaussTest import LinGaussInvProb, LinGaussTest
from helper import get_random_corr_mat

def run_coverage_test(rng, n_reps, m0, C0, Sig, G, Q_true, Q=None):
    """
    The quantities (m0, C0, Sig, G, Q) are fixed throughout
    the whole test. This implies the structure of the inverse problem
        y|u ~ N(Gu, Sig)
        u ~ N(m0, C0)
    is fixed. The quantities (u_true, y, r) are randomized over different
    replications of the experiment. In particular, the ground truth
    parameter is sampled as u_true ~ N(m0, C0) and the synthetic observed
    data is then generated from the model y ~ N(Gu_true, Sig). The
    surrogate bias is sampled as r ~ N(0, Q_true) and the surrogate for
    that replication is then defined as G_star(u) ~ N(Gu + r, Q) with
    r fixed. If Q = Q_true then the surrogate is "calibrated", in the
    sense that Q correctly quantifies the uncertainty in the surrogate
    bias r.
    """

    n = G.shape[0]
    d = G.shape[1]
    tests = []

    # Default to well-calibrated surrogate
    if Q is None:
        Q = Q_true

    # Quantiles to compute for coverage metrics
    probs = np.append(np.arange(0.1, 1.0, step=0.1), 0.99)
    n_probs = len(probs)
    out = {"ep_cover_univariate" : np.empty((n_reps, n_probs, d)),
           "eup_cover_univariate" : np.empty((n_reps, n_probs, d)),
           "ep_cover_joint" : np.empty((n_reps, n_probs)),
           "eup_cover_joint" : np.empty((n_reps, n_probs)),
           "ep_kl" : np.empty(n_reps),
           "eup_kl" : np.empty(n_reps),
           "ep_expected_kl" : np.empty(n_reps),
           "eup_expected_kl" : np.empty(n_reps)}

    for i in range(n_reps):
        inv_prob = LinGaussInvProb(rng, G, m0, C0, Sig)
        r = Gaussian(cov=Q_true, rng=rng).sample()

        test = LinGaussTest(inv_prob, Q, r=r)
        tests.append(test)
        res = test.calc_coverage(probs=probs)

        out["ep_cover_univariate"][i,:,:] = res["ep"]
        out["eup_cover_univariate"][i,:,:] = res["eup"]
        out["ep_cover_joint"][i,:] = test.post.compute_credible_ellipsoid_coverage(test.ep_post)
        out["eup_cover_joint"][i,:] = test.post.compute_credible_ellipsoid_coverage(test.eup_post)
        out["ep_kl"][i] = test.post.kl(test.ep_post)
        out["eup_kl"][i] = test.post.kl(test.eup_post)
        out["ep_expected_kl"][i], out["eup_expected_kl"][i] = test.estimate_expected_kl()

    return tests, out, probs


def plot_coverage(ep_coverage, eup_coverage, probs, q_min=0.05, q_max=0.95, ax=None):
    """
    The first two arguments are shape (n_reps, n_probs), giving the nominal
    coverage for each replicate at each coverage probability level.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ep_m = np.median(ep_coverage, axis=0)
    eup_m = np.median(eup_coverage, axis=0)
    ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)
    eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)

    ax.fill_between(probs, ep_q[0,:], ep_q[1,:], color='blue', alpha=0.3, label='ep')
    ax.fill_between(probs, eup_q[0,:], eup_q[1,:], color='red', alpha=0.3, label='eup')
    ax.plot(probs, ep_m, color='blue', label='ep')
    ax.plot(probs, eup_m, color='red', label='eup')
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color="red", linestyle="--")
    ax.legend()
    plt.close(fig)

    return fig, ax


def plot_coverage_by_dim(ep_coverage, eup_coverage, probs, q_min=0.05,
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

#
# TEMP
#

"""
# Re-run the experiment with smaller Monte Carlo sample to avoid timeouts.
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.stats import chi2
np.random.seed(1)

def kl_gauss(m0,C0,m1,C1):
    d = C0.shape[0]
    invC1 = la.inv(C1)
    term1 = np.trace(invC1 @ C0)
    diff = m1 - m0
    term2 = diff.T @ invC1 @ diff
    term3 = -d
    sign1, logdet1 = np.linalg.slogdet(C1)
    sign0, logdet0 = np.linalg.slogdet(C0)
    term4 = logdet1 - logdet0
    return 0.5*(term1 + term2 + term3 + term4)

d = 3
p = 5
G = np.random.randn(p,d)
Sigma = np.diag(0.5 + np.random.rand(p))
C0 = np.eye(d) * 2.0
m0 = np.zeros(d)
u_true = np.array([0.5, -0.3, 0.8])
r = np.array([0.2, -0.1, 0.0, 0.05, -0.05])

Qs = [0.01, 0.2, 1.0]
results = []
mc_N = 50000

for q_scalar in Qs:
    Q = np.eye(p) * q_scalar
    e_true = np.random.multivariate_normal(np.zeros(p), Q)
    eps = np.random.multivariate_normal(np.zeros(p), Sigma)
    y = G @ u_true + r + e_true + eps

    C = la.inv(G.T @ la.inv(Sigma) @ G + la.inv(C0))
    m = C @ (G.T @ la.inv(Sigma) @ y + la.inv(C0) @ m0)

    Sigma_tilde = Sigma + Q
    y_tilde = y - r
    C1 = la.inv(G.T @ la.inv(Sigma_tilde) @ G + la.inv(C0))
    m1 = C1 @ (G.T @ la.inv(Sigma_tilde) @ y_tilde + la.inv(C0) @ m0)

    H = C @ G.T @ la.inv(Sigma)
    m2 = m - H @ r
    C2 = C + H @ Q @ H.T

    eigs_C = np.linalg.eigvalsh(C)
    eigs_C1 = np.linalg.eigvalsh(C1)
    eigs_C2 = np.linalg.eigvalsh(C2)

    kl_1 = kl_gauss(m, C, m1, C1)
    kl_2 = kl_gauss(m, C, m2, C2)

    us = np.random.multivariate_normal(m, C, size=mc_N)
    invC1 = la.inv(C1)
    invC2 = la.inv(C2)
    qs1 = np.einsum('ni,ij,nj->n', us - m1, invC1, us - m1)
    qs2 = np.einsum('ni,ij,nj->n', us - m2, invC2, us - m2)
    alpha = 0.95
    thresh = chi2.ppf(alpha, df=d)

    cover1 = np.mean(qs1 <= thresh)
    cover2 = np.mean(qs2 <= thresh)

    results.append({
        'q_scalar': q_scalar,
        'eig_C': eigs_C,
        'eig_C1': eigs_C1,
        'eig_C2': eigs_C2,
        'kl_pi_pi1': kl_1,
        'kl_pi_pi2': kl_2,
        'coverage_pi1': cover1,
        'coverage_pi2': cover2,
        'm_diff_norm_pi1': np.linalg.norm(m - m1),
        'm_diff_norm_pi2': np.linalg.norm(m - m2)
    })

rows = []
for rdict in results:
    rows.append({
        'q': rdict['q_scalar'],
        'kl(pi||pi1)': rdict['kl_pi_pi1'],
        'kl(pi||pi2)': rdict['kl_pi_pi2'],
        'coverage(pi1)@95%': rdict['coverage_pi1'],
        'coverage(pi2)@95%': rdict['coverage_pi2'],
        '||m-m1||': rdict['m_diff_norm_pi1'],
        '||m-m2||': rdict['m_diff_norm_pi2'],
        'minEig(C)': np.min(rdict['eig_C']),
        'minEig(C1)': np.min(rdict['eig_C1']),
        'minEig(C2)': np.min(rdict['eig_C2']),
        'maxEig(C)': np.max(rdict['eig_C']),
        'maxEig(C1)': np.max(rdict['eig_C1']),
        'maxEig(C2)': np.max(rdict['eig_C2']),
    })

df = pd.DataFrame(rows).set_index('q')

import caas_jupyter_tools as tools; tools.display_dataframe_to_user("Comparison: KL, coverage, eigenvalues", df)

# Also print the detailed eigenvalues in the notebook output
df_eigs = pd.DataFrame([{
    'q': r['q_scalar'],
    'eig_C': r['eig_C'],
    'eig_C1': r['eig_C1'],
    'eig_C2': r['eig_C2'],
} for r in results])
df_eigs

"""


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
