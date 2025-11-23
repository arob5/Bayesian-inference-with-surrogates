# experiments/linear_Gaussian/inverse_problem_setup.py
"""
Helpers to construct the linear Gaussian deconvolution inverse problem
"""

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

from LinGaussTest import LinGaussInvProb


def make_inverse_problem(rng, d, noise_sd, ker_length, ker_lengthscale, jitter=1e-8):
    grid = np.arange(d)

    # Prior distribution
    m0 = np.zeros(d)
    C0 = gaussian_cov_mat(d, lengthscale=5, scale=1) + jitter * np.identity(d)

    # Forward model (convolution with sampling)
    s = 4 # every sth spatial index is observed
    ker_grid = gaussian_kernel_grid(ker_length, ker_lengthscale)
    G_conv = construct_toeplitz_forward_model(d, ker_grid)

    idx_obs = np.arange(0, d, s)
    n = len(idx_obs)
    H = np.zeros((n, d), dtype=int)
    H[np.arange(n), idx_obs] = 1
    G = H @ G_conv

    # Noise covariance
    variances = np.sqrt(noise_sd)**2 * np.ones(n)
    Sig = np.diag(variances)

    # Construct inverse problem object
    inv_prob = LinGaussInvProb(rng, G, m0, C0, Sig)
    g_conv_true = G_conv @ inv_prob.u_true
    
    return inv_prob, g_conv_true, grid, idx_obs


def plot_exact_post(inv_prob, grid, idx_obs, g_conv_true=None):
    post_sd = np.sqrt(np.diag(inv_prob.post.cov))
    ci_lower = inv_prob.post.mean - 2 * post_sd
    ci_upper = inv_prob.post.mean + 2 * post_sd

    plt.fill_between(grid, ci_lower, ci_upper, color='blue', alpha=0.1, label="+/- 2 post sd")
    plt.plot(grid, inv_prob.u_true, color="black", label="u_true")
    if g_conv_true is not None:
        plt.plot(grid, g_conv_true, color="orange", label="g_true")
    plt.plot(idx_obs, inv_prob.y, "o", color="red", label="y")
    plt.plot(grid, inv_prob.post.mean, color="blue", label="post mean")
    plt.legend()
    plt.show()


def plot_approx_post(test, grid, idx_obs, post_name):
    if post_name == "exact":
        post_rv = test.post
    elif post_name == "eup":
        post_rv = test.eup_post
    elif post_name == "ep":
        post_rv = test.ep_post
    else:
        raise ValueError(f"Invalid post_name {post_name}")

    post_sd = np.sqrt(np.diag(post_rv.cov))
    ci_lower = post_rv.mean - 2 * post_sd
    ci_upper = post_rv.mean + 2 * post_sd

    plt.fill_between(grid, ci_lower, ci_upper, color='blue', alpha=0.1, label="+/- 2 post sd")
    # plt.plot(grid, inv_prob.u_true, color="black", label="u_true")
    # plt.plot(grid, g_conv_true, color="orange", label="g_true")
    plt.plot(idx_obs, test.y, "o", color="red", label="y")
    plt.plot(grid, post_rv.mean, color="blue", label="post mean")
    plt.legend()
    plt.show()












def gaussian_kernel(x, lengthscale):
    return np.exp(-0.5 * (x / lengthscale) ** 2)

def gaussian_kernel_grid(size, lengthscale):
    """
    Evaluates discrete Gaussian kernel exp(-(i/sigma)^2 / 2) at 
    `size` integers. If `size` is odd this will result in symmetry 
    about zero, otherwise it will be off by one. e.g., for `size = 3`
    the values are evaluated at -1, 0, 1. For `size = 4` evaluated 
    at -2, -1, 0, 1. The kernel values are normalized to sum to one.
    """
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / lengthscale) ** 2)
    kernel /= kernel.sum()
    return kernel

def gaussian_cov_mat(d, lengthscale, scale):
    Sig = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1):
            k = gaussian_kernel(i-j, lengthscale)
            Sig[i,j] = k
            Sig[j,i] = k
            
    return scale**2 * Sig

def linear_same_convolution(u, k_vals):
    """
    Linear convolution of `u` with `k_vals`, returning vector of same 
    length as `u`. Discrete convolution of the "same" variety.
    """

    out = np.convolve(u, k_vals, mode="full")

    # Clip so output is of length equal to `len(u)`
    start = len(k_vals) // 2
    end = start + len(u)
    return out[start:end]

def construct_toeplitz_forward_model(d, ker_vals):
    """
    Returns linear matrix G representing the forward model. This is a 
    Toeplitz matrix encoding the discrete linear convolution.
    d is the dimension of the discretized signal u, and ker_vals
    is the vector of discrete kernel evaluations.

    Returns a (d,d) Toeplitz matrix encoding a discrete linear 
    "same" convolution.
    """

    kernel_length = len(ker_vals)

    first_col = np.zeros(d)
    first_col[:kernel_length//2+1] = ker_vals[kernel_length//2::-1]
    
    first_row = np.zeros(d)
    first_row[:kernel_length//2+1] = ker_vals[kernel_length//2:]
    
    G = toeplitz(first_col, first_row)
    return G


def construct_noise_cov(G, variances):
    # Compose covariance matrix in singular basis
    U, s, Vh = np.linalg.svd(G)
    C_eps = (Vh.T * variances) @ Vh

    return C_eps
    


