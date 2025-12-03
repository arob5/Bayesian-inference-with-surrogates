# uncprob/models/vsem/inverse_problem.py
"""
Defines the Bayesian inverse problem for the VSEM experiment.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from typing import Protocol
from numpyro.distributions import Uniform, MultivariateNormal
from scipy.stats import qmc

from uncprop.models.vsem import vsem_jax as vsem
from uncprop.core.inverse_problem import (
    Prior,
    LogLikelihood,
    Posterior,
    Array,
    PRNGKey,
)


class VSEMPrior(Prior):
    """
    The VSEM prior primarily represents the prior distribution over the VSEM 
    parameters that will be calibrated in the experiment. However, it also stores
    `_dists`, which defines the prior over all VSEM parameters. Additional methods
    beyond the standard Prior interface provide access to this extended prior.
    """

    _dists = {
        "kext": Uniform(0.4, 1.0),
        "lar" : Uniform(0.2, 3.0),
        "lue" : Uniform(5e-04, 4e-03),
        "gamma": Uniform(2e-01, 6e-01),
        "tauv": Uniform(5e+02, 3e+03),
        "taus": Uniform(4e+03, 5e+04),
        "taur": Uniform(5e+02, 3e+03), 
        "av": Uniform(0.4, 1.0),
        "veg_init": Uniform(0.0, 10.0), 
        "soil_init": Uniform(0.0, 30.0),
        "root_init": Uniform(0.0, 10.0)
    }

    def __init__(self, par_names: list[str] | None = None):
        """ par_names defines the set of calibration parameters. """
        self._par_names = par_names or vsem.get_vsem_par_names()
        low = jnp.array([dist.low for dist in self.dists.values()])
        high = jnp.array([dist.high for dist in self.dists.values()])
        self._prior_rv = Uniform(low, high)

    @property
    def dists(self):
        return {par: self._dists[par] for par in self._par_names}

    @property
    def dim(self):
        return len(self._par_names)
    
    @property
    def par_names(self):
        return self._par_names
    
    @property
    def support(self):
        return self._prior_rv.low, self._prior_rv.high
    
    def sample(self, key: PRNGKey, n: int = 1):
        return self._prior_rv.sample(key, sample_shape=(n,))

    def log_density(self, x):
        # TODO: Uniform doesnt throw error or return -Inf if value is outside of support.
        #       should maybe manually check for this case.
        
        l = self._prior_rv.log_prob(x)

        # sum over batch axis
        return jnp.sum(l, axis=tuple(range(-1, 0)))
    
    def sample_all_vsem_params(self, key: PRNGKey):
        """
        Return dictionary representing a single sample from all of the parameters, 
        not just the calibration parameters.
        """
        samp = {}
        for par_name, prior in self._dists.items():
            samp[par_name] = prior.sample(key)

        return samp
    
    def sample_lhc(self, key: PRNGKey, n: int = 1):
        """ Latin hypercube sampling """
        rng_key = _numpy_rng_seed_from_jax_key(key)
        rng = np.random.default_rng(seed=rng_key)
        lhc = qmc.LatinHypercube(d=self.dim, rng=rng)

        # Samples unit hypercube [0, 1)^d
        samp = lhc.random(n=n)

        # Scale based on prior bounds
        l, u = self.support
        samp = qmc.scale(samp, l, u)

        return jnp.asarray(samp)


class VSEMLikelihood:
    """
    Defined to conform to the LogDensity protocol. 
    """

    def __init__(self,
                 key: PRNGKey,
                 n_days: int, 
                 par_names: list[str], 
                 ground_truth: dict[str, float] | None = None):
        """
        If provided, `ground_truth` is a dictionary of all VSEM parameters that 
        will be treated as the ground truth. 
        """
        # independent subkeys for simulating driver / sampling noise realization
        key_driver, key_noise = jr.split(key, 2)

        self.n_days = n_days
        self.time_steps, self.driver = vsem.simulate_vsem_driver(key_driver, self.n_days)
        self.par_names = par_names
        self.d = len(par_names)

        all_par_names = vsem.get_vsem_par_names()
        self._par_idx = [all_par_names.index(par) for par in self.par_names]

        # Observation operator defined as monthly averages of LAI.
        self.lai_idx = vsem.get_vsem_output_names().index("lai")
        self.month_start_idx = jnp.arange(start=0, stop=self.n_days, step=31)
        month_stop_idx = jnp.empty_like(self.month_start_idx)
        month_stop_idx = month_stop_idx.at[:-1].set(self.month_start_idx[1:])
        month_stop_idx = month_stop_idx.at[-1].set(self.n_days - 1)
        self.month_stop_idx = month_stop_idx
        self.month_midpoints = jnp.round(0.5 * (self.month_start_idx + self.month_stop_idx))

        # Ground truth
        if ground_truth is None:
            self._all_par_true = {
                "kext": 0.85,
                "lar": 1.86523322e+00,
                "lue": 6.84170991e-04,
                "gamma": 5.04967614e-01,
                "tauv": 2.95868049e+03,
                "taus": 2.58846896e+04,
                "taur": 1.77011520e+03,
                "av": 0.85,
                "veg_init": 3.04573410e+00,
                "soil_init": 2.11415896e+01,
                "root_init": 5.58376223e+00
            }
        else:
            self._all_par_true = ground_truth

        self.par_true = jnp.array([self._all_par_true[par] for par in self.par_names])
        
        # VSEM forward model.
        self.forward_model = vsem.build_vectorized_partial_forward_model(self.driver, self.par_names,
                                                                         par_default=self._all_par_true)

        self.vsem_output_true = self.forward_model(self.par_true)
        self.observable_true = self.obs_op(self.vsem_output_true).flatten()
        self.n = self.observable_true.shape[0]
        self._sigma = 0.1 * jnp.std(self.observable_true)
        self.noise = MultivariateNormal(covariance_matrix=(self._sigma**2) * jnp.identity(self.n))
        self.y = self.observable_true + self.noise.sample(key_noise)
        self._likelihood_rv = MultivariateNormal(loc=self.y, covariance_matrix=self.noise.covariance_matrix)
        
    def __call__(self, x):
        return self.log_density(x)

    def plot_driver(self):
        plt.plot(self.time_steps, self.driver, "o")
        plt.xlabel("days")
        plt.ylabel("PAR")
        plt.show()

    def par_to_obs_map(self, par):
        vsem_output = self.forward_model(par)
        return self.obs_op(vsem_output)

    def obs_op(self, vsem_output):
        """ Observation operator: monthly averages of LAI """
        lai_output = vsem_output[:,:,self.lai_idx]
        monthly_lai_averages = jnp.array(
            [lai_output[:, start:end].mean(axis=1) for start, end in zip(self.month_start_idx, self.month_stop_idx)]
        )        

        return monthly_lai_averages.T
    
    def log_density(self, x):
        pred_obs = self.par_to_obs_map(x)
        return self._likelihood_rv.log_prob(pred_obs)
    
    def log_density_upper_bound(self, x):
        """The log of the determinant term in the Gaussian density. 
        log{det(2*pi*C)^{-1/2}} = -0.5 * d * log(2*pi) - 0.5 * log{det(C)}.

        This term also represents an upper bound on the log density.
        """
        dim_times_two_pi = self.n * jnp.log(2.0 * jnp.pi)
        L = self._likelihood_rv.scale_tril
        log_det_cov = 2.0 * jnp.log(jnp.diag(L)).sum()
        return -0.5 * (dim_times_two_pi + log_det_cov)

    def plot_vsem_outputs(self, par, burn_in_start=0, include_predicted_obs=False):
        output = self.forward_model(par)
        fig, axs = vsem.plot_vsem_outputs(output[:,burn_in_start:,:], nrows=2)

        if include_predicted_obs:
            pred_obs = self.obs_op(output)
            axs[self.lai_idx].plot(self.month_midpoints, pred_obs.T, "o", color="red")

        return fig
    
    def plot_ground_truth(self):
        fig, axs = vsem.plot_vsem_outputs(self.vsem_output_true, nrows=2)

        lai_ax = axs[self.lai_idx]
        lai_ax.plot(self.month_midpoints, self.y, "o", color="red")

        return fig
    
    
def _numpy_rng_seed_from_jax_key(key: PRNGKey) -> int:
    return int(jr.randint(key, (), 0, 2**63 - 1))