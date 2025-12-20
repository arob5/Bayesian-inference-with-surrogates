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
from gpjax.parameters import transform
from numpyro.distributions.transforms import Transform

Bijection = dict


class SingleOutputGPFactory(Protocol):
    def __call__(self, dataset: Dataset) -> tuple[gpx.gps.AbstractPosterior, Bijection]:
        """ Returns a gpjax posterior for a single-output GP"""
        pass

class BatchIndependentGP:
    """ Stores a set of independent gpjax GP posterior objects
    
    Allows for vectorized hyperparameter optimization and prediction.
    Hyperparameters are optimized independently for each GP. This 
    higher level interface creates a batch of independent GPs all of
    the same structure. 
    """
    
    def __init__(self, 
                 posteriors: Sequence[gpx.gps.AbstractPosterior],
                 bijections: Sequence[Bijection],
                 dataset: Dataset):
        """
        Notes:
            dataset is the shared multi-output dataset across all GPs.
        """
        
        if len(posteriors) != len(bijections):
            raise ValueError('posteriors and bijections length mismatch.')

        self.posteriors = posteriors
        self.bijections = bijections
        self.dataset = dataset

    def get_batched_posterior(self):
        return _stack_posteriors(self.posteriors, self.bijections)

    @property
    def out_dim(self):
        return len(self.posteriors)
    

@dataclass
class BatchedPosterior:
    """Container for batched independent GP posteriors."""
    graphdef: object
    params: object
    static_state: tuple
    bijection: Sequence[Bijection]


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
    bijections = []
    
    for i in range(out_dim):
        dataset_i = _get_single_output_dataset(dataset, i)
        posterior_i, bijection_i = gp_factory(dataset_i)
        posteriors.append(posterior_i)
        bijections.append(bijection_i)
    
    return BatchIndependentGP(posteriors, bijections, dataset)


def _get_single_output_dataset(dataset: Dataset, output_idx: int):
    return Dataset(dataset.X, dataset.y[:, [output_idx]])


def _make_batched_loss(
    batched: BatchedPosterior,
    objective,
    dataset: Dataset,
):
    """
    Notes:
    We want to be able to pass a different bijection to each GP.
    bijections are dictionaries so cannot be directly vectorized over,
    so we instead capture the bijection in a closure and vmap over a 
    static index.
    """

    X = dataset.X
    Y = dataset.y
    bijections = batched.bijection  # Python list captured in closure

    def loss(params):
        def single_loss(param, bijection, y):
            param = transform(param, bijection)
            model = nnx.merge(batched.graphdef, param, *batched.static_state)
            return objective(model, Dataset(X, y[:, None]))

        return jax.vmap(single_loss)(
            params,
            bijections,
            jnp.moveaxis(Y, 1, 0), # if Y is 2d, equivalent to Y.T
        )


    # def loss(params):
    #     def single_loss(idx, param):
    #         param = transform(param, bijections[idx])
    #         model = nnx.merge(
    #             batched.graphdef,
    #             param,
    #             *batched.static_state,
    #         )
    #         single_output_dataset = Dataset(X, Y[:, idx:idx+1])
    #         return objective(model, single_output_dataset)

    #     return jax.vmap(single_loss)(jnp.arange(Y.shape[1]), params)

    return loss


def fit_batch_independent_gp(
    batch_gp: BatchIndependentGP,
    objective,
    optim: optax.GradientTransformation,
    *,
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
    batched = batch_gp.get_batched_posterior()
    params_uncstr = _params_to_unconstrained(batched.params, batched.bijection)

    # initialize batch optimizer state and create batch loss
    opt_state = jax.vmap(optim.init)(params_uncstr)
    loss_fn = _make_batched_loss(batched, objective, batch_gp.dataset)

    def step(carry, _):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params)

        updates, opt_state = jax.vmap(optim.update)(
            grads, opt_state, params
        )
        params = jax.vmap(optax.apply_updates)(params, updates)

        carry = (params, opt_state)
        return carry, loss

    # Optimisation loop
    keys = jax.random.split(key, num_iters)
    (params_uncstr, _), history = jax.lax.scan(step, (params_uncstr, opt_state), keys, unroll=unroll)

    # Parameters bijection to constrained space
    params = _params_to_constrained(params_uncstr, batched.bijection)

    # Reconstruct posteriors
    new_posteriors = []
    for param in params:
        model = nnx.merge(batched.graphdef, param, *batched.static_state)
        new_posteriors.append(model)
    new_batch_gp = BatchIndependentGP(new_posteriors, batch_gp.bijections, batch_gp.dataset)

    return new_batch_gp, history


def _params_to_unconstrained(params, bijections):
    """Convert a batch of parameter PyTrees to unconstrained space."""
    unconstrained = [
        transform(param, bij, inverse=True)
        for param, bij in zip(params, bijections)
    ]

    return jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *unconstrained,
    )


def _params_to_constrained(params, bijections):
    """Convert a batch of parameter PyTrees to constrained space."""
    constrained_params = [
        transform(jax.tree_util.tree_map(lambda x: x[i], params), bijections[i])
        for i in range(len(bijections))
    ]

    return constrained_params


def _stack_posteriors(
    posteriors: Sequence[gpx.gps.AbstractPosterior],
    bijections: Sequence[Bijection],
) -> BatchedPosterior:
    """Split and stack independent GP posteriors into a batched PyTree."""
    graphdefs = []
    params = []
    static_states = []

    for posterior, bij in zip(posteriors, bijections):
        graphdef, p, *static = nnx.split(posterior, gpx.parameters.Parameter, ...)
        graphdefs.append(graphdef)
        params.append(p)
        static_states.append(tuple(static))

    # All graphdefs / static states must be identical
    if not all(g == graphdefs[0] for g in graphdefs):
        raise ValueError("All GP posteriors must share the same structure.")

    if not all(s == static_states[0] for s in static_states):
        raise ValueError("All GP static states must be identical.")

    graphdef = graphdefs[0]
    static_state = static_states[0]

    # transpose tree - values of parameter leaves are now batch values
    batched_params = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *params,
    )

    return BatchedPosterior(
        graphdef=graphdef,
        params=batched_params,
        static_state=static_state,
        bijection=bijections,
    )
