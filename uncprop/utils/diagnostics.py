# uncprop/utils/diagnostics.py
"""
MCMC diagnostic utilities shared across experiments.

Functions for computing effective sample size (ESS), Gelman-Rubin R-hat,
and related convergence diagnostics.
"""

import numpy as np


def compute_ess(samples, method='batch_means', batch_size=None):
    """Estimate effective sample size for each parameter.

    Args:
        samples: (n_samples, dim) array of MCMC samples
        method: 'batch_means' or 'autocorrelation'
        batch_size: batch size for batch means method (default: sqrt(n))

    Returns:
        (dim,) array of ESS estimates
    """
    samples = np.array(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    n, dim = samples.shape

    if method == 'batch_means':
        if batch_size is None:
            batch_size = max(1, int(np.sqrt(n)))
        n_batches = n // batch_size
        ess = np.zeros(dim)
        for j in range(dim):
            x = samples[:n_batches * batch_size, j]
            batch_means = x.reshape(n_batches, batch_size).mean(axis=1)
            var_bm = np.var(batch_means, ddof=1)
            var_x = np.var(x, ddof=1)
            if var_bm > 0:
                ess[j] = n * var_x / (batch_size * var_bm)
            else:
                ess[j] = n
        return ess

    elif method == 'autocorrelation':
        ess = np.zeros(dim)
        for j in range(dim):
            x = samples[:, j] - samples[:, j].mean()
            fft = np.fft.fft(x, n=2 * n)
            acf = np.fft.ifft(fft * np.conj(fft)).real[:n]
            acf /= acf[0]

            # Initial monotone sequence estimator (Geyer 1992)
            tau = 1.0
            for k in range(1, n // 2):
                rho_pair = acf[2*k - 1] + acf[2*k]
                if rho_pair < 0:
                    break
                tau += 2 * rho_pair
            ess[j] = n / tau
        return ess

    else:
        raise ValueError(f'Unknown ESS method: {method}')


def compute_rhat(chains):
    """Compute Gelman-Rubin R-hat statistic across multiple chains.

    Args:
        chains: (n_chains, n_samples, dim) array

    Returns:
        (dim,) array of R-hat values. Values near 1.0 indicate convergence.
    """
    chains = np.array(chains)
    m, n, dim = chains.shape

    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    # Between-chain variance
    chain_means = np.mean(chains, axis=1)  # (m, dim)
    B = n * np.var(chain_means, axis=0, ddof=1)

    # Pooled variance estimate
    var_hat = ((n - 1) / n) * W + (1 / n) * B

    rhat = np.sqrt(var_hat / W)
    return rhat
