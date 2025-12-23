# uncprop/models/elliptic_pde/surrogate.py
from __future__ import annotations

from copy import deepcopy
import jax
import jax.random as jr

import optax
import gpjax as gpx
from gpjax import Dataset
from gpjax.parameters import DEFAULT_BIJECTION

from uncprop.custom_types import PRNGKey
from uncprop.core.inverse_problem import Posterior
from uncprop.core.surrogate import construct_design, GPJaxSurrogate
from uncprop.utils.gpjax_models import construct_gp
from uncprop.utils.gpjax_multioutput import (
    BatchedRBF,
    fit_batch_independent_gp,
    get_batch_gp_from_template,
)


def fit_pde_surrogate(key: PRNGKey,
                      posterior: Posterior,
                      n_design: int,
                      design_method: str,
                      gp_train_args: dict | None = None,
                      verbose: bool = True,
                      jitter: float = 0.0,
                      **fit_kwargs):
    """ Top-level function for fitting PDE forward model surrogate

    Note that `posterior` represents the posterior of the exact inverse problem that 
    we are trying to approximate.
    """
    key, key_design = jr.split(key, 2)

    # sample design points
    if design_method == 'lhc':
        prior_sampler = posterior.prior.sample_lhc
    elif design_method == 'uniform':
        prior_sampler = posterior.prior.sample
    else:
        raise ValueError(f'Invalid design method {design_method}')
 
    design = construct_design(key=key_design,
                              design_method=design_method, 
                              n_design=n_design, 
                              prior_sampler=prior_sampler,
                              f=posterior.likelihood.forward_model)
    
    # fit surrogate
    surrogate, batchgp, history = fit_pde_experiment_batch_gp(design, **fit_kwargs)

    return design, surrogate, batchgp, history



def fit_pde_experiment_batch_gp(design: Dataset,
                                learning_rate: float = 0.1,
                                num_iters: int = 1000,
                                bijection: dict = DEFAULT_BIJECTION):
    
    optim = optax.adam(learning_rate)
    objective = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

    def gp_factory(dataset):
        return construct_gp(dataset, set_bounds=False)[0]
    
    # hyperparameter optimization
    batchgp = get_batch_gp_from_template(gp_factory, design)
    batchgp, history = fit_batch_independent_gp(
        batch_gp=batchgp,
        objective=objective,
        optim=optim,
        num_iters=num_iters
    )

    # surrogate model
    surrogate = convert_gp_to_batch_kernel(batchgp.batch_posterior, design)
    
    return surrogate, batchgp, history


def convert_gp_to_batch_kernel(gp: gpx.gps.ConjugatePosterior, design: Dataset):
    gp = deepcopy(gp)
    batch_dim = design.y.shape[1]
    kernel = gp.prior.kernel
    meanf = gp.prior.mean_function
    likelihood = gp.likelihood

    batched_kernel = BatchedRBF(batch_dim=batch_dim,
                                input_dim=design.in_dim,
                                lengthscale=kernel.lengthscale, 
                                variance=kernel.variance)

    batched_prior = gpx.gps.Prior(mean_function=meanf, kernel=batched_kernel)
    batched_posterior = batched_prior * likelihood
    surrogate = GPJaxSurrogate(batched_posterior, design)

    return surrogate