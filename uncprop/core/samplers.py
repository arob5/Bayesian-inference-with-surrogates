# uncprop/core/samplers.py

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro.distributions import MultivariateNormal
import blackjax
from blackjax.mcmc.random_walk import RWInfo
from blackjax.base import (
    UpdateFn,
    Position,
    State,
    Info,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike
from uncprop.core.surrogate import GPJaxSurrogate, SurrogateDistribution
from uncprop.utils.distribution import _sample_gaussian_tril


def mcmc_loop(key: PRNGKey, 
              kernel: UpdateFn, 
              initial_state: State, 
              num_samples: int = 4000):
    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jax.random.split(key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def init_nuts_kernel(key: PRNGKey, 
                     logdensity: Callable,
                     initial_position: Position,
                     num_warmup_steps: int = 1000):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    warmup_key, _ = jax.random.split(key, 2)
    (initial_state, nuts_params), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup_steps)
    kernel = blackjax.nuts(logdensity, **nuts_params).step

    return initial_state, kernel


# -----------------------------------------------------------------------------
# Cut Sampler
# -----------------------------------------------------------------------------

class CutState(NamedTuple):
    """State of the cut sampler chain.

    position
        Current position of the chain.
    proposal_tril
        Lower Cholesky factor of proposal covariance
    logdensity
        Current sampled value of the log density
    """
    position: Array
    proposal_tril: Array
    logdensity: ArrayLike

class CutInfo(NamedTuple):
    """Side information for the cut sampler chain.

    log_ratio
        log of the Metropolis acceptance ratio
    is_accepted
        whether proposal was accepted or not
    """
    log_ratio: ArrayLike
    is_accepted: ArrayLike


def init_cut_kernel(key: PRNGKey,
                    logp_sampler: Callable[[PRNGKey, ArrayLike, ArrayLike], Array],
                    initial_position: Array,
                    u_prop_cov: Array) -> tuple[CutState, Callable]:
    """
    A noisy Metropolis-Hastings (MH) algorithm with a symmetric Gaussian proposal
    with covariance `u_prop_cov`. Proceeds like a typical MH algorithm, but at 
    each iteration the log-density values at the current and proposed points are 
    sampled using `logp_sampler`.
    """

    # build kernel function
    def kernel(key: PRNGKey, state: tuple) -> tuple[CutState, CutInfo]:
        key_proposal, key_lp, key_accept = jr.split(key, 3)

        u_curr, L, _ = state
        u_prop = _sample_gaussian_tril(key_proposal, m=u_curr, L=L).squeeze()

        # sample log-density values at current/proposed points
        lp_curr, lp_prop = logp_sampler(key_lp, u_curr, u_prop).squeeze()

        # u update
        u_next, lp_next, log_alpha, accept = _mh_accept_reject(key_accept,
                                                               lp_curr=lp_curr, lp_prop=lp_prop,
                                                               u_curr=u_curr, u_prop=u_prop)
        
        info = CutInfo(log_ratio=log_alpha, is_accepted=accept)
        next_state = CutState(position=u_next, proposal_tril=L, logdensity=lp_next)

        return next_state, info
    
    # build initial state
    proposal_tril = jnp.linalg.cholesky(u_prop_cov, upper=False)
    initial_logdensity_sample = logp_sampler(key, initial_position, initial_position)[0]
    initial_state = CutState(position=initial_position,
                             proposal_tril=proposal_tril,
                             logdensity=initial_logdensity_sample)

    return initial_state, kernel


# -----------------------------------------------------------------------------
# Preconditioned Crank-Nicolson Random Kernel Algorithm (rk-pcn)
# -----------------------------------------------------------------------------

class RKPCNStateOld(NamedTuple):
    """State of the rkpcn sampler chain.

    position
        Current position of the chain.
    f_at_position
        Current value of f evaluated at current position.
    proposal_tril
        Lower Cholesky factor of proposal covariance for u.
    rho
        Correlation parameter for the pCN proposal.
    logdensity
        Current sampled value of the log density.
    """
    position: Array
    f_at_position: Array
    proposal_tril: Array
    rho: ArrayLike
    logdensity: ArrayLike

class RKPCNInfoOld(NamedTuple):
    """Side information for the cut sampler chain.

    log_ratio
        log of the Metropolis acceptance ratio
    is_accepted
        whether proposal was accepted or not
    """
    log_ratio: ArrayLike
    is_accepted: ArrayLike


def init_rkpcn_kernel_old(key: PRNGKey,
                      log_density: Callable[[Array, Array], float],
                      gp: GPJaxSurrogate,
                      initial_position: Array, 
                      u_prop_cov: Array,
                      pcn_cor: float = 0.99) -> tuple[RKPCNStateOld, Callable]:
    """
    An MCMC sampler to *approximately* sample from the expectation of the random
    measure pi(u; f) with random log density of the form log_p(f(u)), where f ~ GP(m, k).
    Precisely, one seeks to sample from E_f{pi(u; f)}.
     
    The algorithm rk-pcn sampler operates on the extended state space (u, f), which is in
    principle infinite dimensional. In reality, it is only necessary to instantiate 
    bivariate projections of the GP of the form [f(u), f(u')]. Therefore, the practical
    implementation of this algorithm maintains a state of the form (u, f(u)).

    Note that `log_density` and `gp` must be specified so that they can be called like
        key = jax.random.key(3532)
        pred = gp(u) # predictive distribution
        f_pred = pred.sample(key)
        lp_pred = log_density(f_pred, u)

    In other words, `log_density` is parameterized as a function of the surrogate output,
    not as a function of the parameter u. The argument `u` is still passed, as this is 
    useful in certain cases (e.g., constraining the support of u).
    
    The Gaussian predictive distribution must 
    satisfy the following:
        - single input u: gp(u).sample() has shape (p,)
        - multiple inputs u of shape (n,d): gp(u).sample() has shape (n,p)
    """

    # build kernel function
    def kernel(key: PRNGKey, state: tuple) -> tuple[RKPCNStateOld, RKPCNInfoOld]:
        key_jit, key_proposal, key_accept = jr.split(key, 3)

        u, fu, L, rho, _ = state
        v = _sample_gaussian_tril(key_proposal, m=u, L=L).squeeze()

        # just-in-time sample
        fv = gp.condition_then_predict(v, given=(u, fu)).sample(key_jit).squeeze()

        # Projection of law(f) onto (u, v)
        uv = jnp.stack([u, v], axis=0)
        fuv = jnp.stack([fu, fv], axis=0)
        fuv_dist = gp(uv)

        # f update
        guv = _pcn_proposal(key_proposal, fuv, fuv_dist.mean, fuv_dist.chol, rho=rho).squeeze()
        gu = guv[0]
        gv = guv[1]

        ### TEMP
        key, key_temp = jr.split(key_jit, 2)
        gu, _, log_alpha, accept = _mh_accept_reject(key_temp,
                                                     lp_curr=log_density(fu, u), 
                                                     lp_prop=log_density(gu, u),
                                                     u_curr=fu, u_prop=gu)
        gv = jax.lax.cond(accept, lambda _: gv, lambda _: fv, operand=None)
        ###

        # u update
        u_next, lp_next, log_alpha, accept = _mh_accept_reject(key_accept,
                                                               lp_curr=log_density(gu, u), 
                                                               lp_prop=log_density(gv, v),
                                                               u_curr=u, u_prop=v)
        g_u_next = jax.lax.cond(accept, lambda _: gv, lambda _: gu, operand=None)

        info = RKPCNInfoOld(log_ratio=log_alpha, is_accepted=accept)
        next_state = RKPCNStateOld(position=u_next, f_at_position=g_u_next,
                                proposal_tril=L, rho=rho, logdensity=lp_next)

        return next_state, info

    # build initial state
    proposal_tril = jnp.linalg.cholesky(u_prop_cov, upper=False)
    f_at_initial = gp(initial_position).sample(key).squeeze()
    initial_state = RKPCNStateOld(position=initial_position,
                               f_at_position=f_at_initial,
                               proposal_tril=proposal_tril,
                               logdensity=log_density(f_at_initial, initial_position),
                               rho=pcn_cor)

    return initial_state, kernel


# -----------------------------------------------------------------------------
# Preconditioned Crank-Nicolson Random Kernel Algorithm (rk-pcn)
# -----------------------------------------------------------------------------

class RKPCNState(NamedTuple):
    """State of the rkpcn sampler chain.

    position
        Current position of the chain.
    f_at_position
        Current value of f evaluated at current position.
    proposal_tril
        Lower Cholesky factor of proposal covariance for u.
    rho
        Correlation parameter for the pCN proposal.
    logdensity
        Current sampled value of the log density.
    """
    position: Array
    f_position: tuple[Array, Array | None] # (fu, fu_quad)
    logZ: ArrayLike
    quadrature_rule: tuple[Array, Array] # (u_quad, logw_quad)
    proposal_tril: Array
    rho: float
    logdensity: ArrayLike

class RKPCNInfo(NamedTuple):
    """Side information for the cut sampler chain.

    log_ratio
        log of the Metropolis acceptance ratio
    is_accepted
        whether proposal was accepted or not
    """
    log_ratio: ArrayLike
    is_accepted: ArrayLike


def init_rkpcn_kernel(key: PRNGKey,
                      log_density: Callable[[Array, Array], Array],
                      gp: GPJaxSurrogate,
                      initial_position: Array,
                      u_prop_cov: Array,
                      f_update_fn: Callable[..., tuple[RKPCNState, Array]],
                      pcn_cor: float = 0.99) -> tuple[RKPCNState, Callable]:
    """
    An MCMC sampler to *approximately* sample from the expectation of the random
    measure pi(u; f) with random log density of the form log_p(f(u)), where f ~ GP(m, k).
    Precisely, one seeks to sample from E_f{pi(u; f)}.
    
    The algorithm rk-pcn sampler operates on the extended state space (u, f), which is in
    principle infinite dimensional. In reality, it is only necessary to instantiate 
    bivariate projections of the GP of the form [f(u), f(u')]. Therefore, the practical
    implementation of this algorithm maintains a state of the form (u, f(u)).

    Note that `log_density` and `gp` must be specified so that they can be called like
        key = jax.random.key(3532)
        pred = gp(u) # predictive distribution
        f_pred = pred.sample(key)
        lp_pred = log_density(f_pred, u)

    In other words, `log_density` is parameterized as a function of the surrogate output,
    not as a function of the parameter u. The argument `u` is still passed, as this is 
    useful in certain cases (e.g., constraining the support of u).
    
    The Gaussian predictive distribution must 
    satisfy the following:
        - single input u: gp(u).sample() has shape (p,)
        - multiple inputs u of shape (n,d): gp(u).sample() has shape (n,p)

    Notes:
        The notation used in the function body is as follows:
            - f/g: current/proposed GP trajectories
            - u/v: current/proposed parameter positions
            - unext/fnext: next state
        We use the notation fv to mean the value of f at input v, and so on.
    """

    # build kernel function
    def kernel(key: PRNGKey, state: RKPCNState) -> tuple[RKPCNState, RKPCNInfo]:

        key_u_proposal, key_f_update, key_u_accept = jr.split(key, 2)

        u = state.position
        fu = state.f_position

        # Propose new u position
        L = state.proposal_tril
        v = _sample_gaussian_tril(key_u_proposal, m=u, L=L).squeeze()

        # f update
        state, fnext_v = f_update_fn(key=key_f_update, 
                                     state=state,
                                     gp=gp,
                                     log_density=log_density,
                                     u_prop=v)
        fnext_u = state.f_position

        # u update
        u_next, lp_next, log_alpha, accept = _mh_accept_reject(key_u_accept,
                                                               lp_curr=state.logdensity, 
                                                               lp_prop=log_density(fnext_v, v),
                                                               u_curr=u, u_prop=v)
        fnext_unext = jax.lax.cond(accept, lambda _: fnext_v, lambda _: fnext_u, operand=None)

        # Updated state and auxiliary info
        info = RKPCNInfo(log_ratio=log_alpha, is_accepted=accept)
        next_state = RKPCNState(position=u_next,
                                f_position=(g_u_next, gU),
                                logZ=logZg,
                                quadrature_rule=(U, logw),
                                proposal_tril=L, 
                                rho=rho, 
                                logdensity=lp_next)

        return next_state, info

    #
    # build initial state
    #
    proposal_tril = jnp.linalg.cholesky(u_prop_cov, upper=False)
    key_init, key_quad = jr.split(key, 2)

    # initial position
    f_initial_position = gp(initial_position).sample(key_init).squeeze()
    lp_init = log_density(f_initial_position, initial_position)

    # quadrature points
    U, logw = quadrature_rule
    U = jnp.atleast_2d(U)
    logw = logw.ravel()
    fU = gp(U).sample(key_quad).squeeze()
    lp_quad = log_density(fU, U)
    logZf = logZ_approx(lp_quad, logw)

    initial_state = RKPCNState(position=initial_position,
                               f_position=(f_initial_position, fU),
                               logZ=logZf,
                               quadrature_rule=(U, logw),
                               proposal_tril=proposal_tril, 
                               rho=pcn_cor, 
                               logdensity=lp_init)

    return initial_state, kernel


def _f_update_pcn_proposal(key: PRNGKey,
                           *,
                           state: RKPCNState,
                           gp: GPJaxSurrogate,
                           log_density: Callable[[Array, Array], Array], 
                           u_prop: Array,
                           **kwargs):
    """ pCN proposal without a Metropolis correction. 
    
    Realizes pCN proposal at the finite set of points (u, u_prop), where u
    is the current position.
    """

    rho = state.f_update_info.rho
    u = state.position
    fu = state.f_position

    gU, _ = _just_in_time_pcn_proposal(key=key,
                                       gp=gp,
                                       given=(u, fu),
                                       u_jit=u_prop,
                                       rho=rho)

    # Update state
    gu = gU[0]
    gv = gU[1]
    state = state._replace(f_position=gu, logdensity=log_density(gu, u))

    return state, gv


def _f_update_eup_cpm(key: PRNGKey,
                      *,
                      state: RKPCNState,
                      gp: GPJaxSurrogate, 
                      log_density: Callable[[Array, Array], Array],
                      u_prop: Array,
                      **kwargs):
    """ Correlation pseudo-marginal (cPM) update for targeting expected unnormalized posterior (EUP) 
    
    pCN proposal with a Metropolis correction that uses ratio pi(u; g)/pi(u; f) of unnormalized 
    densities. Does not include normalizing constants Z(f)/Z(g) in this ratio, which makes this
    update target the EUP, not the EP.
    """
    key_proposal, key_accept = jr.split(key, 2)

    rho = state.f_update_info.rho
    u = state.position
    fu = state.f_position
    lu = state.logdensity

    # Proposal
    gU, fv = _just_in_time_pcn_proposal(key=key_proposal,
                                        gp=gp,
                                        given=(u, fu),
                                        u_jit=u_prop,
                                        rho=rho)
    gu = gU[0]
    gv = gU[1]

    # Accept-Reject with EUP target
    fnext_u, lnext_u, log_alpha, accept = _mh_accept_reject(key_accept,
                                                            lp_curr=lu, 
                                                            lp_prop=log_density(gu, u),
                                                            u_curr=fu, u_prop=gu)
    fnext_v = jax.lax.cond(accept, lambda _: gv, lambda _: fv, operand=None)

    # Update state
    state = state._replace(f_position=fnext_u, logdensity=lnext_u)

    return state, fnext_v



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mh_accept_reject(key: PRNGKey,
                      lp_curr: float,
                      lp_prop: float,
                      u_curr: ArrayLike,
                      u_prop: ArrayLike,
                      log_correction: ArrayLike = 0.0) -> tuple[Array, Array, Array, Array]:
    """ Assumes symmetric proposal """
    log_unif = jnp.log(jr.uniform(key))

    log_alpha = lp_prop - lp_curr + log_correction
    log_alpha = jnp.where(jnp.isnan(log_alpha), -jnp.inf, log_alpha)
    accept = log_unif < log_alpha

    u_next, lp_next = jax.lax.cond(
        accept,
        lambda _: (u_prop, lp_prop),
        lambda _: (u_curr, lp_curr),
        operand=None,
    )   

    return u_next, lp_next, log_alpha, accept


def _pcn_proposal(key: PRNGKey,
                  x: Array,
                  mean: Array, 
                  cov_tril: Array, 
                  rho: float) -> Array:
    pcn_mean = mean + rho * (x - mean)
    pcn_tril = jnp.sqrt(1 - rho**2) * cov_tril

    return _sample_gaussian_tril(key, m=pcn_mean, L=pcn_tril)


def _just_in_time_pcn_proposal(key: PRNGKey,
                               *,
                               gp: GPJaxSurrogate,
                               given: tuple[Array, Array],
                               u_jit: Array,
                               rho: float):
    """ Just-in-time sample then pCN proposal.

    Given a realization `given = (u, fu)` of a GP at inputs u, and another set
    of inputs `u_jit`. First, just-in-time samples values at `u_jit` via
    f_jit ~ law(f(u_jit) | f(u) = u). Then returns a pCN proposal at the 
    combined set of inputs (u, u_jit), given current value (fu, f_jit).

    Returns:
        tuple:
            fU_prop: the pCN proposal at inputs (u, u_jit)
            f_jit: the just in time sample f_jit at input u_jit
    """

    key_jit, key_proposal = jr.split(key, 2)

    # just-in-time sample
    f_jit = gp.condition_then_predict(u_jit, given=given).sample(key_jit).squeeze()

    # pCN proposal, realized at points (u, u_jit)
    u, fu = given
    U = jnp.vstack([u, u_jit])
    fU = jnp.concatenate([jnp.reshape(fu, -1), jnp.reshape(f_jit, -1)])
    fU_dist = gp(U)
    fU_prop = _pcn_proposal(key_proposal, fU, fU_dist.mean, fU_dist.chol, rho=rho).squeeze()

    return fU_prop, f_jit


def _logZ_approx(logw: ArrayLike, lp_U: ArrayLike, axis=None):
    return logsumexp(logw + lp_U, axis=axis)


def stack_dict_arrays(data_dict: dict[str, ArrayLike], 
                      names: list[str] | None = None) -> Array:
    """
    Given a dictionary of arrays, each of shape (n,) or (n,d) (with d allowed to vary
    by array), concatenate to array of shape (n, d_total). The arrays are concatenated
    left-to-right in the order specified by `names`.
    """
    if names is None:
        names = list(data_dict.keys())

    arrays = [jnp.asarray(data_dict[name]) for name in names]
    arrays = jax.tree.map(tree=arrays, f=_ensure_2d)

    n_rows = arrays[0].shape[0]
    assert all(a.shape[0] == n_rows for a in arrays), "All arrays must have same number of rows"
    return jnp.concatenate(arrays, axis=1) 


def get_trace_plots(trace: dict[str, Array], 
                    names: list[str] | None = None,
                    num_rows: int = 1,
                    figsize: tuple = (15, 6)):

    if names is None:
        names = list(trace.keys())

    num_plots = len(names)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_plots-num_rows+1)
    if num_plots > 1:
        axs = axs.ravel()

    for i in range(num_plots):
        ax = axs[i]
        nm = names[i]
        trajectory = trace[nm]

        ax.plot(trajectory)
        ax.set_xlabel('samples')
        ax.set_ylabel(nm)
    
    return fig, axs


def _ensure_2d(x):
    x = jnp.asarray(x)
    assert x.ndim <= 2

    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x



if __name__ == '__main__':
    import jax.scipy.stats as stats
    from datetime import date

    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    loc, scale = 10, 20
    observed = np.random.normal(loc, scale, size=1_000)

    def logdensity_fn(loc, log_scale, observed=observed):
        """Univariate Normal"""
        scale = jnp.exp(log_scale)
        logjac = log_scale
        logpdf = stats.norm.logpdf(observed, loc, scale)
        return logjac + jnp.sum(logpdf)
    logdensity = lambda x: logdensity_fn(**x)

    initial_position = {"loc": 1.0, "log_scale": 1.0}

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)

    kernel = blackjax.nuts(logdensity, **parameters).step
    
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, 1_000)

    mcmc_samples = states.position
    mcmc_samples["scale"] = jnp.exp(mcmc_samples["log_scale"]).block_until_ready()

    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
    ax.plot(mcmc_samples["loc"])
    ax.set_xlabel("Samples")
    ax.set_ylabel("loc")

    ax1.plot(mcmc_samples["scale"])
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("scale")
    fig.savefig('/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper/uncprop/test.png')


