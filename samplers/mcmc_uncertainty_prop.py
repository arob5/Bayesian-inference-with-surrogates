# mcmc_uncertainty_prop.py
"""
MCMC algorithms for uncertainty propagation in surrogate-based inverse problems
"""
from __future__ import annotations
import numpy as np

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.samplers import MCMCSampler
from typing import Any, Callable, Protocol, runtime_checkable, TypeVar

from modmcmc.kernels import (
    MarkovKernel,
    MetropolisKernel,
    GaussianProposal,
    GaussMetropolisKernel, 
    DiscretePCNProposal,
    DiscretePCNKernel, 
    UncalibratedDiscretePCNKernel, 
    mvn_logpdf
)

# -----------------------------------------------------------------------------
# Protocols for working with Gaussian Processes
# -----------------------------------------------------------------------------

Array = np.typing.NDArray

@runtime_checkable
class DistWithMeanCov(Protocol):
    """
    Protocol for distribution-like objects that expose
    `.mean` and `.cov` attributes/properties which are Arrays.
    """
    mean: Array
    cov: Array

    def sample(self, n: int = 1) -> Array:
        ...

@runtime_checkable
class GaussianProcess(Protocol):
    """
    Protocol for a callable GP-like object:

        f(x: Array, given: Optional[Tuple[Array, Array]] = None) -> DistWithMeanCov

    Where `given` (if provided) is a tuple (x_new, y_new). Here (x_new, y_new) is a new dataset
    on which the GP is conditioned. The call returns a object representing the multivariate
    Gaussian obtained by projecting the (potentially conditional) GP onto the finite set
    of input points `x`.
    """
    def __call__(self, x: Array, given: tuple[Array, Array] | None = None) -> DistWithMeanCov:
        ...


# -----------------------------------------------------------------------------
# MCMC algorithms for uncertainty propagation in surrogate-based inverse problems
# -----------------------------------------------------------------------------


def get_mwg_eup_sampler(self, u_prop_scale=0.1, pcn_cor=0.99):
    """
    Exactly targets the EUP.
    """
    L_noise = self.noise.chol

    # Extended state space. Initialize state via prior sample.
    state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

    # Target density.
    def ldens_post(state):
        fwd = self.G @ state.primary["u"] + state.primary["e"]
        return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

    target = TargetDensity(LogDensityTerm("post", ldens_post))

    # u and e updates.
    ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                    term_subset="post", block_vars="u", rng=self.rng)
    ker_e = DiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                cor_param=pcn_cor, term_subset="post",
                                block_vars="e", rng=self.rng)

    # Sampler
    alg = BlockMCMCSampler(target, initial_state=state,
                            kernels=[ker_u, ker_e], rng=self.rng)
    return alg


def get_rk_sampler(inv_prob, u_prop_scale=0.1):
    L_noise = self.noise.chol

    # Initialize state via prior sample.
    state = State(primary={"u": self.prior.sample()})

    # Noisy target density.
    def ldens_post_noisy(state):
        fwd = self.G @ state.primary["u"] + self.e.sample()
        return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

    target = TargetDensity(LogDensityTerm("post", ldens_post_noisy), use_cache=False)

    # Metropolis-Hastings updates.
    ker = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov, rng=self.rng)

    # Sampler
    alg = BlockMCMCSampler(target, initial_state=state, kernels=ker, rng=self.rng)
    return alg


def get_rk_pcn_sampler(self, u_prop_scale=0.1, pcn_cor=0.9):
    L_noise = self.noise.chol

    # Extended state space. Initialize state via prior sample.
    state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

    # Target density.
    def ldens_post(state):
        fwd = self.G @ state.primary["u"] + state.primary["e"]
        return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])
    target = TargetDensity(LogDensityTerm("post", ldens_post))

    # u and e updates.
    ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                  term_subset="post", block_vars="u", rng=self.rng)
    ker_e = UncalibratedDiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                          cor_param=pcn_cor, block_vars="e", rng=self.rng)

    # Sampler
    alg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e], rng=self.rng)
    return alg


# TODO: currently assumes symmetric proposal
class RandomKernelPCNSampler(MCMCSampler):
    """
    An MCMC sampler to *approximately* sample from the expectation of the random
    measure pi(u; f) with random log density of the form log_p(f(u)), where f ~ GP(m, k).
    Precisely, one seeks to sample from E_f{pi(u; f)}.
     
    The algorithm rk-pcn sampler operates on the extended state space (u, f), which is in
    principle infinite dimensional. In reality, it is only necessary to instantiate 
    bivariate projections of the GP of the form [f(u), f(u')]. Therefore, the practical
    implementation of this algorithm has state space of the form [u, u', f(u), f(u')].
    The first slot is the actual state we are ultimately interested in; the other 
    three are auxiliary variables.
    """
    def __init__(self,
                 log_density: Callable[[Array], float],
                 gp: GaussianProcess,
                 u_init: Array,
                 u_prop_cov: Array | None = None,
                 pcn_cor: float = 0.99,
                 rng = None):
        
        # Extended state space.
        u_dim = u_init.flatten().shape[0]
        fu_init = gp(u_init.reshape(1,-1)).mean.squeeze()
        init_state = State(primary={"u": u_init, "fu": fu_init})

        def target_log_density(state):
            return log_density(state.primary["fu"])
        target = TargetDensity(LogDensityTerm("density_given_f", target_log_density))
        super().__init__(target=target, initial_state=init_state, rng=rng)

        # Proposal distribution for u updates.
        # u_prop_cov = u_prop_cov or np.identity(u_dim)
        # self.u_proposal = GaussianProposal(u_prop_cov, rng=self.rng)
        self.ker_u = _RandomKernelPCNHelper(base_ker_u, log_density, gp, rng)

        # Data for f updates.
        self.pcn_cor = pcn_cor
        self.gp = gp
        
    def step(self) -> State:
        # This algorithm requires 
        #  (1) a MH kernel over u
        #  (2) a PCN kernel wrt to a bivariate Gaussian [f, f']
        #
        # To clarify notation, let u, f denote current values of u and f, so that fu is the 
        # current value of f(u). Let v, g be proposed values. So fv is the current value of 
        # f evaluated at the proposed value of u.

        current_state = self.current_state.copy_with(clear_cache=False)

        # Propose new u value so that the f update knows which bivariate projection to use
        u_block = current_state.extract_block(var_names="u")
        u = u_block["u"]
        fu = current_state.extract_block(var_names="fu")["fu"]
        v = self.ker_u.proposal.propose(u_block)["u"]
        # v = self.u_proposal.propose(u_block)["u"]
        uv = np.vstack([u, v])

        # Just in time sample for f(v)
        fv = self.gp(v, given=(u, fu)).sample()
        fuv = np.concatenate([fu, fv])

        # f update
        fuv_dist = self.gp(uv)
        guv = DiscretePCNProposal(mean=fuv_dist.mean, cov=fuv_dist.cov, cor_param=self.pcn_cor, rng=self.rng).propose({"fuv": fuv})["fuv"] # [g(u), g(v)]

        # u update: note that proposal already occurred above
        new_state = self.ker_u(guv) # {u, fu}, (gu, gv); MH ratio uses (gu, gv)

        # so proposal should return 


        # u = _finish_mh_step(fuv, guv)

        # return new_state


def _finish_mh_step(uv, guv, target):
    u, v = uv[0], uv[1]
    gu, gv = guv[0], guv[1]
    gu_state = State(primary={"u": u_init, "fu": fu_init})


    log_p_u, log_p_v = target(gu), 







class _RandomKernelPCNHelper(MetropolisKernel):
    def __init__(self,
                 base_kernel: MetropolisKernel,
                 log_density: Callable[[Array], float],
                 gp: GaussianProcess,
                 rng = None):

        self._check_base_kernel(base_kernel)

        # Define target log density
        def target_log_density(state):
            return log_density(state.primary["fu"])
        target = TargetDensity(LogDensityTerm("density_given_f", target_log_density))


        init_state = State(primary={"u": u_init, "fu": fu_init})

        super().__init__(target=target, is_exact=False, rng=rng)


    def step(self, state: State, v) -> State:
        
        v["fu"] = 

        candidate = state.copy_with(primary_updates=v) # updates u

        # Compute MH log‐acceptance ratio.
        log_p_old = self.target(state, term_names=self.term_subset)
        log_p_new = self.target(candidate, term_names=self.term_subset)

        if self.acc_ratio_includes_prop:
            log_q_fwd = self.proposal.log_p(new_block, old_block)
            log_q_back = self.proposal.log_p(old_block, new_block)
        else:
            log_q_fwd, log_q_back = 0,0

        log_alpha = (log_p_new - log_p_old) + (log_q_back - log_q_fwd)
        info = {"log_alpha": log_alpha}

        # Accept/reject step.
        if np.log(self.rng.random()) < log_alpha:
            info["accepted"] = True
            return candidate
        else:
            info["accepted"] = False
            return state
        

    def _check_base_kernel(self, base_kernel):
        if not isinstance(base_kernel, MetropolisKernel):
            raise TypeError(f"_RandomKernelPCNHelper requires base_kernel to be a MetropolisKernel. Got {type(base_kernel)}")

    


    


def _finish_mh_step(u, v, fu, fv, log_density, rng):
    log_ratio = log_density(fv) - log_density(fu)
    
    # Accept/reject step.
    if np.log(rng.random()) < log_ratio:
        return v, fv
    else:
        return u, fu


    # Propose new state with updated block.
    old_block = state.extract_block(var_names=self.block_vars)
    new_block = self.proposal.propose(old_block)
    candidate = state.copy_with(primary_updates=new_block)

    # Compute MH log‐acceptance ratio.
    log_p_old = self.target(state, term_names=self.term_subset)
    log_p_new = self.target(candidate, term_names=self.term_subset)

    if self.acc_ratio_includes_prop:
        log_q_fwd = self.proposal.log_p(new_block, old_block)
        log_q_back = self.proposal.log_p(old_block, new_block)
    else:
        log_q_fwd, log_q_back = 0,0

    log_alpha = (log_p_new - log_p_old) + (log_q_back - log_q_fwd)
    info = {"log_alpha": log_alpha}

    # Accept/reject step.
    if np.log(self.rng.random()) < log_alpha:
        info["accepted"] = True
        return candidate
    else:
        info["accepted"] = False
        return state