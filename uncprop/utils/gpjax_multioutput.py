# uncprop/utils/gpjax_multioutput.py
"""
Helpers for vectorized batch independent multioutput GPs.
"""
from __future__ import annotations
from typing import Protocol
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from flax import nnx
import jax
import jax.numpy as jnp
from jax import vmap, jit

import optax
import gpjax as gpx
from gpjax import Dataset
from gpjax.parameters import transform, DEFAULT_BIJECTION
from gpjax.gps import AbstractPosterior
from numpyro.distributions.transforms import Transform

Bijection = dict


class SingleOutputGPFactory(Protocol):
    def __call__(self, dataset: Dataset) -> AbstractPosterior:
        """ Returns a gpjax posterior for a single-output GP"""
        pass


class BatchIndependentGP:
    """ Stores a set of independent gpjax GP posterior objects
    
    This class provides the functionality to map between two different 
    representations of the batch GP:
        (1) a list of single-output GP posteriors
        (2) a single batch GP posterior

    The batch GP posterior is a tree in which the parameter leaves have an
    added batch dimension (the first dimension). This is the representation
    that allows for vectorized hyperparameter optimization and prediction.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 posterior_list: Sequence[AbstractPosterior] | None = None,
                 batch_posterior: AbstractPosterior | None = None):
        """
        Notes:
            dataset is the shared multi-output dataset across all GPs.
        """
        if not ((posterior_list is None) ^ (batch_posterior is None)):
            raise ValueError('BatchIndependentGP requires exactly one of posterior_list or batch_posterior.')
        
        self.dataset = dataset
        self.dim_out = dataset.y.shape[1]

        if(posterior_list is None):
            self.posterior_list, self.tree_info = _posterior_batch_to_list(batch_posterior, self.dim_out)
            self.batch_posterior = batch_posterior
        else:
            self.batch_posterior, self.tree_info = _posterior_list_to_batch(posterior_list)
            self.posterior_list = posterior_list

    @property
    def graphdef(self):
        return self.tree_info[0]

    @property
    def params(self):
        return self.tree_info[1]
    
    @property
    def static_state(self):
        return self.tree_info[2]


def get_batch_gp_from_template(gp_factory: SingleOutputGPFactory,
                               dataset: Dataset) -> BatchIndependentGP:
    """
    Args:
        gp_factory: callable that returns one single-output GP posterior
        dataset: gpjax Dataset, with "y" of shape (N, Q) where Q is the number
                    of GPs in the batch.
    """
    out_dim = dataset.y.shape[1]
    posteriors = []
    
    for i in range(out_dim):
        dataset_i = _get_single_output_dataset(dataset, i)
        posterior_i = gp_factory(dataset_i)
        posteriors.append(posterior_i)
    
    return BatchIndependentGP(dataset=dataset, posterior_list=posteriors)


def _get_single_output_dataset(dataset: Dataset, output_idx: int):
    return Dataset(dataset.X, dataset.y[:, [output_idx]])


def _make_batched_loss_and_grad(batch_gp: BatchIndependentGP,
                                objective,
                                bijection: Bijection,
                                dataset: Dataset):
    X = dataset.X
    Y = dataset.y
    graphdef = batch_gp.graphdef
    static = batch_gp.static_state

    # loss/grad for a single-output GP
    def single_loss(param, y):
        param_constr = transform(param, bijection)
        model = nnx.merge(graphdef, param_constr, *static)
        return objective(model, Dataset(X, y[:, None]))
    
    single_grad = jax.grad(single_loss, argnums=0)

    # batched independent loss/gradient computation
    loss_vect = jax.vmap(single_loss, in_axes=(0, 1))
    grad_vect = jax.vmap(single_grad, in_axes=(0, 1))
                         
    def loss_and_grad(params):
        return loss_vect(params, Y), grad_vect(params, Y)

    return loss_and_grad


def fit_batch_independent_gp(
    batch_gp: BatchIndependentGP,
    objective,
    optim: optax.GradientTransformation,
    *,
    bijection: Bijection = DEFAULT_BIJECTION,
    num_iters: int = 100,
    key: jax.Array = jax.random.PRNGKey(0),
    unroll: int = 1,
):
    """
    Fit a batch of independent GP models using vectorized optimisation.

    Args:
        batch_gp: BatchIndependentGP instance.
        objective: Scalar loss function (e.g. negative log marginal likelihood).
        optim: Optax optimizer.
        num_iters: Number of optimisation steps.
        key: PRNGKey.
        unroll: Scan unroll factor.

    Returns:
        Updated BatchIndependentGP with trained posteriors.
        Training loss history of shape (num_iters, Q).
    """
    params_unconstr = transform(batch_gp.params, bijection, inverse=True)

    # initialize batch optimizer state and create batch loss
    opt_state = jax.vmap(optim.init)(params_unconstr)
    loss_and_grad_fn = _make_batched_loss_and_grad(batch_gp, objective, bijection, batch_gp.dataset)

    @jit
    def step(carry, _):
        params, opt_state = carry
        loss, grads = loss_and_grad_fn(params)
        updates, opt_state = jax.vmap(optim.update)(grads, opt_state, params)
        params = jax.vmap(optax.apply_updates)(params, updates)
        carry = (params, opt_state)
        return carry, loss

    # Optimisation loop
    keys = jax.random.split(key, num_iters)
    (params_unconstr, _), history = jax.lax.scan(step, (params_unconstr, opt_state), keys, unroll=unroll)

    # Parameters bijection to constrained space
    params = transform(params_unconstr, bijection)

    # Return batch posterior with updated parameters
    new_tree = nnx.merge(batch_gp.graphdef, params, *batch_gp.static_state)
    new_batch_gp = BatchIndependentGP(dataset=batch_gp.dataset, batch_posterior=new_tree)

    return new_batch_gp, history


def _posterior_batch_to_list(posterior_batch, dim_out) -> tuple[Sequence[AbstractPosterior], tuple]:
    """
    dim_out is the number of outputs in the multioutput GP (i.e., the length of
    the batch dimension).
    """

    graphdef, params, *static = nnx.split(posterior_batch, gpx.parameters.Parameter, ...)

    posterior_list = []
    for i in range(dim_out):
        param = jax.tree.map(lambda p: p[i], params)
        post = nnx.merge(graphdef, param, *static)
        posterior_list.append(post)

    info = (graphdef, params, static)
    return posterior_list, info


def _posterior_list_to_batch(
        posterior_list: Sequence[AbstractPosterior]
) -> tuple[AbstractPosterior, tuple]:
    """Split and stack independent GP posteriors into a batched PyTree.
    
    Returns:
        tuple:
            - batch posterior: gpjax AbstractPosterior with batch parameters
            - info: tuple with graphdef, batch_params, static that can be combined to 
                    form the batch posterior using nnx.merge.
    """
    graphdefs = []
    params = []
    static_states = []

    for post in posterior_list:
        graphdef, param, *static = nnx.split(post, gpx.parameters.Parameter, ...)
        graphdefs.append(graphdef)
        params.append(param)
        static_states.append(tuple(static))

    # All graphdefs / static states must be identical
    if not all(g == graphdefs[0] for g in graphdefs):
        raise ValueError("All GP posteriors must share the same structure.")
    if not all(s == static_states[0] for s in static_states):
        raise ValueError("All GP static states must be identical.")
    
    graphdef = graphdefs[0]
    static_state = static_states[0]

    # transpose tree - values of parameter leaves are now batch values
    batch_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params)
    batch_post = nnx.merge(graphdef, batch_params, *static_state)
    info = (graphdef, batch_params, static_state)
    
    return batch_post, info