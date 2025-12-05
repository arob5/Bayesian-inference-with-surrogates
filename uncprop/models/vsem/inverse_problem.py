# uncprob/models/vsem/inverse_problem.py
'''
Defines the Bayesian inverse problem for the VSEM experiment.
'''
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from typing import Protocol
from numpyro.distributions import Uniform, MultivariateNormal
from scipy.stats import qmc

from uncprop.custom_types import Array, PRNGKey
from uncprop.models.vsem import vsem_jax as vsem
from uncprop.core.inverse_problem import (
    Prior,
    LogLikelihood,
    Posterior,
)

VSEM_DEFAULT_PARAMS = {
    'kext': 0.85,
    'lar': 1.86523322e+00,
    'lue': 6.84170991e-04,
    'gamma': 5.04967614e-01,
    'tauv': 2.95868049e+03,
    'taus': 2.58846896e+04,
    'taur': 1.77011520e+03,
    'av': 0.85,
    'veg_init': 3.04573410e+00,
    'soil_init': 2.11415896e+01,
    'root_init': 5.58376223e+00
}


VSEM_DEFAULT_PRIORS = {
    'kext': Uniform(0.4, 1.0),
    'lar' : Uniform(0.2, 3.0),
    'lue' : Uniform(5e-04, 4e-03),
    'gamma': Uniform(2e-01, 6e-01),
    'tauv': Uniform(5e+02, 3e+03),
    'taus': Uniform(4e+03, 5e+04),
    'taur': Uniform(5e+02, 3e+03), 
    'av': Uniform(0.4, 1.0),
    'veg_init': Uniform(0.0, 10.0), 
    'soil_init': Uniform(0.0, 30.0),
    'root_init': Uniform(0.0, 10.0)
}


@dataclass
class ObservationOperatorInfo:
    """Stores the observation operator and intermediate quantities
    
    The intermediate quantities define the structure of the observation operator,
    and are primarily helpful for plotting. The observation operator is assumed
    to be of the form "compute window averages of one of the VSEM outputs"
    (e.g., monthly averages of LAI).
    """
    observation_operator: Callable
    window_mask: Array
    window_lens: Array
    output_idx: int


@dataclass
class DataGeneratingProcess:
    """Ground truth data generating process.
    
    Assumes an additive Gaussian noise model. `noise_cov_tril` is the lower 
    Cholesky factor of the Gaussian noise.
    """
    driver_key: PRNGKey
    driver: Array
    vsem_params: dict[str, float]
    observation_operator_info: ObservationOperatorInfo
    noise_cov_tril: Array


@dataclass
class CalibrationModel:
    """The model used for the data generating process when calibrating parameters
    
    This includes specification of calibration parameters (VSEM parameters that will
    be learned from data) and "fixed parameters". `vsem_params` defines the defaults
    for all VSEM parameters - the parameters specified in `calibration_params` will
    override the defaults, and the remaining parameters will be fixed at the specified
    default values. The `forward_model`, which is a map from the calibration parameters to
    VSEM outputs, is build by `build_vectorized_partial_forward_model()`. 
    The observation operator is a map from VSEM outputs to an observable quantity.
    """
    driver: Array
    vsem_params: dict[str, float]
    calibration_params: list[str]
    observation_operator_info: ObservationOperatorInfo
    noise_cov_tril: Array

    def __post_init__(self):
        forward_model = vsem.build_vectorized_partial_forward_model(driver=self.driver, 
                                                                    par_names=self.calibration_params, 
                                                                    par_default=self.vsem_params)
        self.forward_model = forward_model
        
        def param_to_observable_map(x):
            vsem_output = self.forward_model(x)
            observable = self.observation_operator_info.observation_operator(vsem_output)
            return observable
        
        self.param_to_observable_map = param_to_observable_map


@dataclass
class DataRealization:
    """Stores observed data and intermediate quantities
    
    Intermediate quantities are the VSEM forward model output and the observable
    (output of the observation operator before adding noise).
    """
    obs_key: PRNGKey
    data_generating_process: DataGeneratingProcess
    forward_output: Array
    observable: Array
    observation: Array

     # calibration parameters
   #  all_param_names = vsem.get_vsem_par_names()
   #  param_idx = [all_param_names.index(par) for par in par_names]   


def obs_op_window_means(vsem_output: Array,
                        output_idx: int,
                        window_mask: Array,
                        window_lens: Array):
    """ Observation operator: window averages of LAI """

    lai_output = vsem_output[:,:,output_idx].T # (n_days, n_ensemble)
    lai_window_sums = window_mask @ lai_output # (n_windows, n_ensemble)
    lai_window_means = lai_window_sums.T / window_lens # (n_ensemble, n_windows) 

    return lai_window_means


def define_vsem_observation_operator(num_days: int, 
                                     window_len: int = 30,
                                     vsem_output_var: str = 'lai'):
    """Constructs a ObservationOperatorInfo using the obs_op_window_means observation operator"""

    # create windows for averaging
    window_start_idx = jnp.arange(0, num_days, window_len)
    window_stop_idx = window_start_idx + window_len
    n_windows = len(window_start_idx)

    if int(window_stop_idx[-1]) != int(num_days):
        raise ValueError(f'last entry of window_stop_idx should equal n_days = {num_days}.' 
                         f' Got {window_stop_idx[-1]}.')

    # mask of shape (n_windows, n_days); ones mark days in window
    window_mask = jnp.vstack(
        [jnp.concatenate([jnp.zeros(start), jnp.ones(end-start), jnp.zeros(num_days-end)])
         for start, end in zip(window_start_idx, window_stop_idx)]
    )
    window_lens = jnp.sum(window_mask, axis=1)

    # Observation operator defined as windowly averages of an VSEM output variable.
    vsem_output_idx = vsem.get_vsem_output_names().index(vsem_output_var)

    # JAX compatible closure
    def obs_op(vsem_output):
        return obs_op_window_means(vsem_output=vsem_output,
                                   output_idx=vsem_output_idx,
                                   window_mask=window_mask,
                                   window_lens=window_lens)

    return ObservationOperatorInfo(
        observation_operator=obs_op,
        window_mask=window_mask,
        window_lens=window_lens,
        output_idx=vsem_output_idx
    )


def define_vsem_likelihood(driver: Array,
                           par_names: list[str],
                           noise_cov: Array,
                           window_len: int = 30):
    '''
    Returns data defining the likelihood model. The likelihood is assumed
    to be of the form N(y | G(u), noise_cov), where G is the composition
    of the VSEM forward model with an observation operator. The observation
    operator maps to window means of leaf area index (LAI) (default montly means).
    '''
    
    # create windows for averaging
    n_days = len(driver)
    window_start_idx = jnp.arange(0, n_days, window_len)
    window_stop_idx = window_start_idx + window_len
    n_windows = len(window_start_idx)

    if int(window_stop_idx[-1]) != int(n_days):
        raise ValueError(f'last entry of window_stop_idx should equal n_days = {n_days}.' 
                         f' Got {window_stop_idx[-1]}.')
    if len(window_start_idx) != noise_cov.shape[0]:
        raise ValueError(f'noise_cov dimension should equal number of windows = {n_windows}'
                         f' Got {noise_cov.shape[0]}.')

    # mask of shape (n_windows, n_days); ones mark days in window
    window_mask = jnp.vstack(
        [jnp.concatenate([jnp.zeros(start), jnp.ones(end-start), jnp.zeros(n_days-end)])
         for start, end in zip(window_start_idx, window_stop_idx)]
    )
    window_lens = jnp.sum(window_mask, axis=1)

    # Observation operator defined as windowly averages of LAI.
    vsem_output_idx = vsem.get_vsem_output_names().index('lai')

    # Lower Cholesky factor of noise covariance



    return VSEMLikelihoodInfo(driver=driver,
                              par_names=par_names,
                              window_mask=window_mask,
                              window_lens=window_lens,
                              output_idx=vsem_output_idx,
                              noise_cov_tril=noise_cov)

    driver_seed: PRNGKey
    driver: Array
    vsem_params: dict[str, float]
    forward_model: Callable
    observation_operator: Callable
    noise_cov_tril: Array



def simulate_vsem_ground_truth(key: PRNGKey,
                               driver: Array,
                               noise_cov_tril: Array,
                               vsem_true_params: dict[str, float] = VSEM_DEFAULT_PARAMS):

    pass



# def simulate_observations_with_intermediates(key: PRNGKey,
#                                              param: Array,
#                                              forward_model: Callable,
#                                              obs_op: Callable,
#                                              driver: Array,
#                                              noise_cov_tril: Array):
#     param = jnp.asarray(param).ravel()
#     vsem_output = forward_model(param)
#     observable = obs_op(vsem_output)
#     noise_realization = jr.normal(key, shape=())

#     observations = noise_cov_tril.


class VSEMPrior(Prior):
    '''
    The VSEM prior primarily represents the prior distribution over the VSEM 
    parameters that will be calibrated in the experiment. However, it also stores
    `_dists`, which defines the prior over all VSEM parameters. Additional methods
    beyond the standard Prior interface provide access to this extended prior.
    '''

    # dictionary of marginals for all VSEM parameters
    _dists = VSEM_DEFAULT_PRIORS

    def __init__(self, par_names: list[str] | None = None):
        ''' par_names defines the set of calibration parameters. '''
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
        '''
        Return dictionary representing a single sample from all of the parameters, 
        not just the calibration parameters.
        '''
        samp = {}
        for par_name, prior in self._dists.items():
            samp[par_name] = prior.sample(key)

        return samp
    
    def sample_lhc(self, key: PRNGKey, n: int = 1):
        ''' Latin hypercube sampling '''
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
    '''
    Defined to conform to the LogDensity protocol. 
    '''

    def __init__(self,
                 key: PRNGKey,
                 time_steps: Array,
                 driver: Array, 
                 par_names: list[str], 
                 ground_truth: dict[str, float] | None = None):
        '''
        If provided, `ground_truth` is a dictionary of all VSEM parameters that 
        will be treated as the ground truth. 
        '''
        # independent subkeys for simulating driver / sampling noise realization
        key_driver, key_noise = jr.split(key, 2)

        time_steps = jnp.asarray(time_steps).ravel()
        driver = jnp.asarray(driver).ravel()
    
        n_days = time_steps.shape[0]
        if driver.shape[0] != n_days:
            raise ValueError(f'time_steps and driver length mismatch: {time_steps[0]} vs {driver[0]}')

        self.time_steps, self.driver = vsem.simulate_vsem_driver(key_driver, self.n_days)

        self.n_days = n_days
        self.driver = driver
        self.time_steps = time_steps
        self.par_names = par_names
        self.d = len(par_names)

        all_par_names = vsem.get_vsem_par_names()
        self._par_idx = [all_par_names.index(par) for par in self.par_names]

        # Observation operator defined as windowly averages of LAI.
        self.lai_idx = vsem.get_vsem_output_names().index('lai')
        self.window_start_idx = jnp.arange(start=0, stop=self.n_days, step=31)
        window_stop_idx = jnp.empty_like(self.window_start_idx)
        window_stop_idx = window_stop_idx.at[:-1].set(self.window_start_idx[1:])
        window_stop_idx = window_stop_idx.at[-1].set(self.n_days - 1)
        self.window_stop_idx = window_stop_idx
        self.window_midpoints = jnp.round(0.5 * (self.window_start_idx + self.window_stop_idx))

        # Ground truth
        if ground_truth is None:
            self._all_par_true = {
                'kext': 0.85,
                'lar': 1.86523322e+00,
                'lue': 6.84170991e-04,
                'gamma': 5.04967614e-01,
                'tauv': 2.95868049e+03,
                'taus': 2.58846896e+04,
                'taur': 1.77011520e+03,
                'av': 0.85,
                'veg_init': 3.04573410e+00,
                'soil_init': 2.11415896e+01,
                'root_init': 5.58376223e+00
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
        plt.plot(self.time_steps, self.driver, 'o')
        plt.xlabel('days')
        plt.ylabel('PAR')
        plt.show()

    def par_to_obs_map(self, par):
        vsem_output = self.forward_model(par)
        return self.obs_op(vsem_output)

    def obs_op(self, vsem_output):
        ''' Observation operator: windowly averages of LAI '''
        lai_output = vsem_output[:,:,self.lai_idx]
        windowly_lai_averages = jnp.array(
            [lai_output[:, start:end].mean(axis=1) for start, end in zip(self.window_start_idx, self.window_stop_idx)]
        )        

        return windowly_lai_averages.T
    
    def log_density(self, x):
        pred_obs = self.par_to_obs_map(x)
        return self._likelihood_rv.log_prob(pred_obs)
    
    def log_density_upper_bound(self, x):
        '''The log of the determinant term in the Gaussian density. 
        log{det(2*pi*C)^{-1/2}} = -0.5 * d * log(2*pi) - 0.5 * log{det(C)}.

        This term also represents an upper bound on the log density.
        '''
        dim_times_two_pi = self.n * jnp.log(2.0 * jnp.pi)
        L = self._likelihood_rv.scale_tril
        log_det_cov = 2.0 * jnp.log(jnp.diag(L)).sum()
        return -0.5 * (dim_times_two_pi + log_det_cov)

    def plot_vsem_outputs(self, par, burn_in_start=0, include_predicted_obs=False):
        output = self.forward_model(par)
        fig, axs = vsem.plot_vsem_outputs(output[:,burn_in_start:,:], nrows=2)

        if include_predicted_obs:
            pred_obs = self.obs_op(output)
            axs[self.lai_idx].plot(self.window_midpoints, pred_obs.T, 'o', color='red')

        return fig
    
    def plot_ground_truth(self):
        fig, axs = vsem.plot_vsem_outputs(self.vsem_output_true, nrows=2)

        lai_ax = axs[self.lai_idx]
        lai_ax.plot(self.window_midpoints, self.y, 'o', color='red')

        return fig
    
    
def _numpy_rng_seed_from_jax_key(key: PRNGKey) -> int:
    return int(jr.randint(key, (), 0, 2**63 - 1))