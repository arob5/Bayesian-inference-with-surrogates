'''
vsem_jax.py
JAX implementation of the Very Simple Ecosystem Model (VSEM).

See BayesianTools R package:
    https://search.r-project.org/CRAN/refmans/BayesianTools/html/VSEM.html
'''

from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence, Callable, Mapping

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util, lax
import matplotlib.pyplot as plt

from uncprop.custom_types import Array, PRNGKey


# ---------------------------------------------------------------------
# Canonical names - defines canonical ordering
# ---------------------------------------------------------------------

canonical_par_names = [
    'kext',      # KEXT - light extinction coefficient
    'lar',       # LAR  - leaf area ratio
    'lue',       # LUE  - light use efficiency
    'gamma',     # GAMMA - autotrophic respiration fraction
    'tauv',      # tauV - veg longevity
    'taus',      # tauS - soil longevity
    'taur',      # tauR - root longevity
    'av',        # Av - fraction of NPP to veg
    'veg_init',  # Cv initial
    'soil_init', # Cs initial
    'root_init'  # Cr initial
]

canonical_output_names = ['veg', 'soil', 'root', 'nee', 'lai']


# ---------------------------------------------------------------------
# Pytree dataclasses for VSEM inputs
# ---------------------------------------------------------------------

@tree_util.register_pytree_node_class
@dataclass
class VSEMParam:
    """Container for VSEM parameters. Registered as a JAX pytree."""
    kext: float | Array
    lar: float | Array
    lue: float | Array
    gamma: float | Array
    tauv: float | Array
    taus: float | Array
    taur: float | Array
    av: float | Array
    veg_init: float | Array
    soil_init: float | Array
    root_init: float | Array

    def tree_flatten(self):
        children = (self.kext, self.lar, self.lue, self.gamma, self.tauv,
                    self.taus, self.taur, self.av, self.veg_init,
                    self.soil_init, self.root_init)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class VSEMInitialCondition:
    """Initial conditions container (veg_init, soil_init, root_init).
    
    TODO: at present the API for running this model does not differentiate
    between trait parameters and initial conditions - they are all considered
    "parameters". This class doesn't have much use beyond the conceptual.
    Eventually, should update so that the interface is defined in terms of 
    VSEMInput, a pytree with subtrees for driver/params/IC. For batch inputs,
    should maintain the pytree structure rather than flattening.
    """
    veg_init: float | Array
    soil_init: float | Array
    root_init: float | Array

    def tree_flatten(self):
        children = (self.veg_init, self.soil_init, self.root_init)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class VSEMInput:
    """Top-level pytree container for inputs required by the solver."""
    param: VSEMParam
    initial_condition: VSEMInitialCondition
    driver: Array

    def tree_flatten(self):
        children = (self.param, self.initial_condition, self.driver)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ---------------------------------------------------------------------
# Small helpers - models for leaf area index / net ecosystem exchange
# ---------------------------------------------------------------------

def compute_lai(xv: Array, lar: float | Array) -> Array:
    """LAI = LAR * xv."""
    return lar * xv


def compute_nee(xv: Array,
                driver: Array,
                gamma: float | Array,
                lue: float | Array,
                kext: float | Array,
                lar: float | Array) -> Array:
    """Compute NEE(t) = (1-gamma) * lue * PAR(t) * (1 - exp(-kext * LAI))."""
    lai = compute_lai(xv, lar)
    return (1.0 - gamma) * lue * driver * (1.0 - jnp.exp(-kext * lai))


# ---------------------------------------------------------------------
# ODE and solver (compatible with JAX transforms)
# ---------------------------------------------------------------------

def vsem_rhs(x: Array, par: VSEMParam, driver_t: float | Array) -> Array:
    """RHS of VSEM ODE at a single time step."""
    xv, xs, xr = x[0], x[1], x[2]

    # Unpack parameters
    k = par.kext
    lar = par.lar
    lue = par.lue
    gamma = par.gamma
    tauv = par.tauv
    taus = par.taus
    taur = par.taur
    av = par.av

    npp = compute_nee(xv, driver_t, gamma, lue, k, lar)

    dxv_dt = av * npp - xv / tauv
    dxs_dt = xr / taur + xv / tauv - xs / taus
    dxr_dt = (1.0 - av) * npp - xr / taur

    return jnp.stack([dxv_dt, dxs_dt, dxr_dt])


def _vsem_step(carry: Array,
               t_and_driver: tuple[int, float | Array],
               par: VSEMParam) -> tuple[Array, Array]:
    """Single forward-Euler step for use with lax.scan."""
    _, driver = t_and_driver
    x_prev = carry
    dx = vsem_rhs(x_prev, par, driver)
    x_next = x_prev + dx  # daily time step

    veg = x_next[0]
    soil = x_next[1]
    root = x_next[2]
    nee = compute_nee(xv=veg, driver=driver, gamma=par.gamma, lue=par.lue,
                      kext=par.kext, lar=par.lar)
    lai = compute_lai(xv=veg, lar=par.lar)

    output_row = jnp.stack([veg, soil, root, nee, lai])
    return x_next, output_row


@jax.jit
def solve_vsem(vsem_input: VSEMInput) -> Array:
    """Solve VSEM with forward Euler using lax.scan.

    Returns:
        model_out: (n_timesteps, 5) with columns [veg, soil, root, nee, lai]
    """
    driver = vsem_input.driver
    n_timesteps = driver.shape[0]

    x0 = jnp.array([
        vsem_input.initial_condition.veg_init,
        vsem_input.initial_condition.soil_init,
        vsem_input.initial_condition.root_init
    ])

    # initial outputs correspond to initial condition
    param = vsem_input.param
    init_nee = compute_nee(xv=x0[0], driver=driver[0], gamma=param.gamma, lue=param.lue,
                           kext=param.kext, lar=param.lar)
    init_lai = compute_lai(xv=x0[0], lar=param.lar)
    init_row = jnp.stack([x0[0], x0[1], x0[2], init_nee, init_lai])

    # if driver has only a single time step, just return initial condition
    is_single_step = jnp.equal(n_timesteps, 1)
    def single_branch(_):
        return init_row[None, :]

    def multi_branch(_):
        driver_tail = driver[1:]
        idx_tail = jnp.arange(1, n_timesteps)
        seq_tail = (idx_tail, driver_tail)
        carry0 = x0
        step_fn = lambda carry, t_and_driver: _vsem_step(carry, t_and_driver, vsem_input.param)
        _, outputs_tail = lax.scan(step_fn, carry0, seq_tail)
        model_out = jnp.vstack([init_row, outputs_tail])
        return model_out

    return lax.cond(is_single_step, single_branch, multi_branch, operand=None)


# ---------------------------------------------------------------------
# Defaults and conversions between named dicts and canonical vectors
# ---------------------------------------------------------------------

def get_vsem_default_pars_dict() -> dict[str, float]:
    """Return canonical default parameter values as a dict keyed by canonical_par_names."""
    return {
        'kext': 0.5,
        'lar': 1.5,
        'lue': 0.002,
        'gamma': 0.4,
        'tauv': 1440.0,
        'taus': 27370.0,
        'taur': 1440.0,
        'av': 0.5,
        'veg_init': 3.0,
        'soil_init': 15.0,
        'root_init': 3.0
    }


def canonical_defaults_array(par_default: Mapping[str, float] | None = None) -> Array:
    """Return canonical default parameter vector in canonical order (jnp.array)."""
    d = par_default or get_vsem_default_pars_dict()
    return jnp.asarray([d[name] for name in canonical_par_names])


def par_vector_from_named_dict(named: Mapping[str, Any],
                               par_default: Mapping[str, float] | None = None) -> VSEMParam:
    """Build a VSEMParam pytree from a Python mapping name->value (user-facing)."""
    defaults = par_default or get_vsem_default_pars_dict()
    vals = []
    for name in canonical_par_names:
        if name in named:
            vals.append(jnp.asarray(named[name]))
        else:
            vals.append(jnp.asarray(defaults[name]))
    return VSEMParam(*vals)


def make_vsem_input_from_named(par_named: Mapping[str, Any],
                               driver: Sequence[float] | Array,
                               par_default: Mapping[str, float] | None = None) -> VSEMInput:
    """Convenience: construct a VSEMInput from named parameter mapping"""
    param = par_vector_from_named_dict(par_named, par_default)
    initial_condition = VSEMInitialCondition(veg_init=param.veg_init,
                                             soil_init=param.soil_init,
                                             root_init=param.root_init)
    driver_jnp = jnp.asarray(driver)
    return VSEMInput(param=param, initial_condition=initial_condition, driver=driver_jnp)


def vector_to_vsemparam(full_vector: Array) -> VSEMParam:
    """Convert a full-length canonical-order vector to a VSEMParam pytree.

    This function is JAX-traceable. full_vector must have shape 
    (len(canonical_par_names),).
    """
    return VSEMParam(*tuple(full_vector[i] for i in range(len(canonical_par_names))))


# ---------------------------------------------------------------------
# Forward model builders:
#   Vectorized VSEM runs as function of varying input parameters
# ---------------------------------------------------------------------

def build_batch_forward_model(driver: Array,
                              par_names: list[str],
                              default_param_dict: Mapping[str, float] | None = None):
    """
    Returns forward_batch(batch_of_partial_params) -> batched outputs.
    - driver is treated as fixed (captured in closure) and should be a jnp array.
    - par_names are validated against canonical_par_names.
    - default_param_dict is converted to canonical defaults array.
    """

    # convert driver once (do NOT convert to tuple)
    driver = jnp.asarray(driver)

    # Validate parameter names
    for name in par_names:
        if name not in canonical_par_names:
            raise ValueError(f"Unknown parameter name: {name}")

    # indices as a small python tuple (static) OR as a jnp array precomputed:
    indices = tuple(canonical_par_names.index(name) for name in par_names)
    indices_arr = jnp.asarray(indices, dtype=jnp.int32)   # used in-array ops

    # default_param as a DeviceArray (captured in closure)
    default_param = canonical_defaults_array(default_param_dict)

    def _single_run(partial_param: Array):
        param_vec = default_param.at[indices_arr].set(partial_param)
        param = vector_to_vsemparam(param_vec)

        initial_condition = VSEMInitialCondition(veg_init=param.veg_init,
                                                 soil_init=param.soil_init,
                                                 root_init=param.root_init)
        vsem_input = VSEMInput(param=param,
                               initial_condition=initial_condition,
                               driver=driver) # driver is array captured in closure
        return solve_vsem(vsem_input)

    # vectorize over the first axis of a batch of partial params, then jit the batched function
    forward_batch = jax.jit(jax.vmap(_single_run, in_axes=0, out_axes=0))

    return forward_batch


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def get_vsem_par_names() -> list[str]:
    """Return canonical parameter names."""
    return list(canonical_par_names)


def get_vsem_output_names() -> list[str]:
    """Return canonical output names."""
    return list(canonical_output_names)


def get_default_prior_bounds() -> dict[str, tuple[float, float]]:
    """Uniform prior bounds for canonical parameters."""
    return {
        "kext": (0.2, 1.0),
        "lar": (0.2, 3.0),
        "lue": (5e-4, 4e-3),
        "gamma": (0.2, 0.6),
        "tauv": (5e2, 3e3),
        "taus": (4e3, 5e4),
        "taur": (5e2, 3e3),
        "av": (2e-1, 1.0),
        "veg_init": (0.0, 10.0),
        "soil_init": (0.0, 30.0),
        "root_init": (0.0, 10.0)
    }


def simulate_vsem_driver(key: PRNGKey, n_days: int) -> tuple[Array, Array]:
    """Generate a synthetic PAR time series (time_steps, driver)."""
    time_steps = jnp.arange(n_days)
    driver = 10 * jnp.abs(jnp.sin(time_steps / 365 * jnp.pi) + 0.25 * jr.normal(key, (n_days,)))
    return time_steps, driver


def plot_vsem_outputs(output: Array,
                      output_names: Sequence[str] | None = None,
                      nrows: int = 1,
                      ncols: int | None = None,
                      figsize: tuple[float, float] = (5, 4),
                      plot_kwargs: dict | None = None):
    """Plot model outputs. `output` expected (n_run, n_time, n_output) or (n_time, n_output)."""
    if output.ndim == 2:
        output = output[jnp.newaxis, ...]

    if output_names is None:
        output_names = get_vsem_output_names()
    if plot_kwargs is None:
        plot_kwargs = {}

    n_plots = len(output_names)
    if ncols is None:
        ncols = int(np.ceil(n_plots / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).flatten()
    time_steps = np.arange(output.shape[1])
    all_output_names = get_vsem_output_names()

    for j in range(n_plots):
        output_name = output_names[j]
        output_idx = all_output_names.index(output_name)
        ax = axs[j]
        ax.plot(time_steps, output[:, :, output_idx].T, **plot_kwargs)
        ax.set_title(output_name)
        ax.set_xlabel('days')
        ax.set_ylabel(output_name)

    for k in range(n_plots, nrows*ncols):
        fig.delaxes(axs[k])
    plt.close(fig)
    return fig, axs








# #
# # OLD
# #


# # -----------------------------------------------------------------------------
# # Canonical parameter / output names
# # -----------------------------------------------------------------------------

# canonical_par_names = [
#     'kext',      # KEXT - light extinction coefficient
#     'lar',       # LAR  - leaf area ratio
#     'lue',       # LUE  - light use efficiency
#     'gamma',     # GAMMA - autotrophic respiration fraction
#     'tauv',      # tauV - veg longevity
#     'taus',      # tauS - soil longevity
#     'taur',      # tauR - root longevity
#     'av',        # Av - fraction of NPP to veg
#     'veg_init',  # Cv initial
#     'soil_init', # Cs initial
#     'root_init'  # Cr initial
# ]

# canonical_output_names = ['veg', 'soil', 'root', 'nee', 'lai']

# # -----------------------------------------------------------------------------
# # Pytree classes: Parameter set, Initial conditions, Driver, and wrapper
# #   for all VSEM inputs
# # -----------------------------------------------------------------------------

# @tree_util.register_pytree_node_class
# @dataclass
# class VSEMParam:
#     '''Container for VSEM parameters. Registered as a JAX pytree.

#     Attributes
#     ----------
#     kext, lar, lue, gamma, tauv, taus, taur, av, veg_init, soil_init, root_init
#         All stored as floats or jnp arrays (broadcastable).
#     '''
#     kext: float | Array
#     lar: float | Array
#     lue: float | Array
#     gamma: float | Array
#     tauv: float | Array
#     taus: float | Array
#     taur: float | Array
#     av: float | Array
#     veg_init: float | Array
#     soil_init: float | Array
#     root_init: float | Array

#     # pytree flatten/unflatten
#     def tree_flatten(self):
#         children = (self.kext, self.lar, self.lue, self.gamma, self.tauv,
#                     self.taus, self.taur, self.av, self.veg_init,
#                     self.soil_init, self.root_init)
#         aux_data = None
#         return children, aux_data

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)


# @tree_util.register_pytree_node_class
# @dataclass
# class VSEMInitialCondition:
#     '''Container for initial conditions (veg_init, soil_init, root_init).'''
#     veg_init: float | Array
#     soil_init: float | Array
#     root_init: float | Array

#     def tree_flatten(self):
#         children = (self.veg_init, self.soil_init, self.root_init)
#         return children, None

#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         return cls(*children)


# @tree_util.register_pytree_node_class
# @dataclass
# class VSEMInput:
#     '''Top-level pytree container for VSEM inputs.

#     Attributes
#     ----------
#     param : VSEMParam
#         Model parameters
#     initial_condition : VSEMInitialCondition
#         Initial conditions (veg_init, soil_init, root_init)
#     driver : jnp.ndarray
#         1D array of photosynthetically active radiation (PAR) time series (length = number of days)
#     '''
#     param: VSEMParam
#     initial_condition: VSEMInitialCondition
#     driver: Array

#     def tree_flatten(self):
#         children = (self.param, self.initial_condition, self.driver)
#         return children, None

#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         return cls(*children)


# # ---------------------------------------------------------------------
# # Small helper functions (LAI, NEE)
# # ---------------------------------------------------------------------

# def compute_lai(xv: Array, lar: float | Array) -> Array:
#     '''Compute LAI = LAR * xv.

#     xv may be a scalar or array (time series).
#     '''
#     return lar * xv


# def compute_nee(xv: Array,
#                 par_driver: Array,
#                 gamma: float | Array,
#                 lue: float | Array,
#                 kext: float | Array,
#                 lar: float | Array) -> Array:
#     '''Compute NEE: (1-gamma) * lue * PAR * (1 - exp(-k * LAI)).'''
#     lai = compute_lai(xv, lar)
#     return (1.0 - gamma) * lue * par_driver * (1.0 - jnp.exp(-kext * lai))


# # -----------------------------------------------------------------------------
# # Core model equations (righthand side vector field) and step function
# # -----------------------------------------------------------------------------

# def vsem_rhs(x: Array,
#              par: VSEMParam,
#              par_driver_t: float | Array) -> Array:
#     '''RHS of VSEM ODE at a single time step.

#     Parameters
#     ----------
#     x : array-like, shape (3,)  -> [xv, xs, xr]
#     par : VSEMParam (pytree)
#     par_driver_t : scalar PAR at time t

#     Returns
#     -------
#     dx_dt : jnp.ndarray shape (3,)
#     '''
#     xv, xs, xr = x[0], x[1], x[2]

#     # Unpack parameters
#     k = par.kext
#     lar = par.lar
#     lue = par.lue
#     gamma = par.gamma
#     tauv = par.tauv
#     taus = par.taus
#     taur = par.taur
#     av = par.av

#     npp = compute_nee(xv, par_driver_t, gamma, lue, k, lar)

#     dxv_dt = av * npp - xv / tauv
#     dxs_dt = xr / taur + xv / tauv - xs / taus
#     dxr_dt = (1.0 - av) * npp - xr / taur

#     return jnp.stack([dxv_dt, dxs_dt, dxr_dt])


# def vsem_step(carry: Array,
#               t_and_driver: tuple[int, float | Array],
#               par: VSEMParam) -> tuple[Array, Array]:
#     '''Single Euler time-step for the VSEM. Designed for use within lax.scan.

#     carry: previous state (x_{t-1})
#     t_and_driver: (t_index, par_driver[t])
#     Returns: (new_carry, output_row) where output_row contains [veg, soil, root, nee, lai]
#     '''
#     _, par_driver = t_and_driver
#     x_prev = carry
#     dx = vsem_rhs(x_prev, par, par_driver)
#     x_next = x_prev + dx  # forward Euler (dt = 1 day)

#     # Compute outputs
#     veg = x_next[0]
#     soil = x_next[1]
#     root = x_next[2]
#     nee = compute_nee(veg, par_driver, par.gamma, par.lue, par.kext, par.lar)
#     lai = compute_lai(veg, par.lar)
#     output_row = jnp.stack([veg, soil, root, nee, lai])
#     return x_next, output_row


# # ---------------------------------------------------------------------
# # Solve using lax.scan
# # ---------------------------------------------------------------------

# @jax.jit
# def solve_vsem_jax(vsem_input: VSEMInput) -> Array:
#     '''Solve the VSEM ODE with forward Euler using lax.scan.

#     Parameters
#     ----------
#     vsem_input : VSEMInput
#         Contains param (VSEMParam), initial_condition (VSEMInitialCondition),
#         and driver (1D jnp array of PAR values).

#     Returns
#     -------
#     model_out : jnp.ndarray shape (n_time, 5) with columns (veg, soil, root, nee, lai)
#     '''
#     driver = vsem_input.driver
#     n_timesteps = driver.shape[0]

#     # initial state from vsem_input.initial_condition
#     x0 = jnp.array([vsem_input.initial_condition.veg_init,
#                     vsem_input.initial_condition.soil_init,
#                     vsem_input.initial_condition.root_init])
    
#     # Create sequence for scan: indices are not used except to match original interface
#     seq = (jnp.arange(n_timesteps), driver)

#     # First row of output will correspond to initial condition.
#     init_nee = compute_nee(x0[0], driver[0], vsem_input.param.gamma,
#                            vsem_input.param.lue, vsem_input.param.kext, vsem_input.param.lar)
#     init_lai = compute_lai(x0[0], vsem_input.param.lar)
#     init_row = jnp.stack([x0[0], x0[1], x0[2], init_nee, init_lai])

#     # define scanning sequence for t = 1, ..., n-1
#     if n_timesteps == 1:
#         return init_row[None, :]
#     else:
#         driver_tail = driver[1:]
#         idx_tail = jnp.arange(1, n_timesteps)
#         seq_tail = (idx_tail, driver_tail)
#         carry0 = x0
#         step_fn = lambda carry, t_and_driver: vsem_step(carry, t_and_driver, vsem_input.param)
#         carry_final, outputs_tail = lax.scan(step_fn, carry0, seq_tail)
#         model_out = jnp.vstack([init_row, outputs_tail])
#         return model_out


# # -----------------------------------------------------------------------------
# # Forward model (vectorized) & utilities for partial parameter specification
# # ----------------------------------------------------------------------------

# def par_vector_from_named_dict(named: Mapping[str, Any],
#                                par_default: Mapping[str, float] | None = None) -> VSEMParam:
#     '''Build a VSEMParam from a dict mapping names -> values.

#     Accepts a dict with a subset of canonical_par_names; missing entries are filled
#     from par_default (a mapping name->float). If par_default is None, use the
#     canonical defaults from get_default_pars().
#     '''
#     if par_default is None:
#         par_default = get_vsem_default_pars_dict()

#     # Defensive: accept any order; case-sensitive
#     vals = {}
#     for name in canonical_par_names:
#         if name in named:
#             vals[name] = jnp.asarray(named[name])
#         else:
#             vals[name] = jnp.asarray(par_default[name])
#     return VSEMParam(*[vals[name] for name in canonical_par_names])


# def get_vsem_default_pars_dict() -> dict:
#     '''Return canonical default parameter values as a dict keyed by canonical_par_names.'''
#     defaults = {
#         'kext': 0.5,
#         'lar': 1.5,
#         'lue': 0.002,
#         'gamma': 0.4,
#         'tauv': 1440.0,
#         'taus': 27370.0,
#         'taur': 1440.0,
#         'av': 0.5,
#         'veg_init': 3.0,
#         'soil_init': 15.0,
#         'root_init': 3.0
#     }
#     return defaults


# def make_vsem_input_from_named(par_named: Mapping[str, Any],
#                                driver: Sequence[float] | Array,
#                                par_default: Mapping[str, float] | None = None) -> VSEMInput:
#     '''Convenience to create a VSEMInput from a (possibly partial) named parameter dict.'''
#     param = par_vector_from_named_dict(par_named, par_default)
#     initial = VSEMInitialCondition(param.veg_init, param.soil_init, param.root_init)
#     driver_jnp = jnp.asarray(driver)
#     return VSEMInput(param=param, initial_condition=initial, driver=driver_jnp)


# def build_forward_model(driver: Sequence[float] | Array,
#                         par_default: Mapping[str, float] | None = None,
#                         output_names: Sequence[str] | None = None) -> Callable:
#     '''Return a function `fwd(par_subset_named)` that evaluates VSEM for named parameter sets.

#     The returned `fwd` accepts:
#       - a single dict mapping parameter-name -> scalar (for a single run), or
#       - a sequence/array of dicts (for multiple runs), or
#       - an (n_runs, n_params) array together with a matching list-of-names (see build_forward_model_named_vmap).
#     For vectorized runs it's often easiest to use `build_forward_model_named_vmap` (below).

#     This function primarily helps for single-run convenience.
#     '''
#     par_default_dict = par_default or get_vsem_default_pars_dict()
#     driver_jnp = jnp.asarray(driver)

#     def fwd_single(par_named: Mapping[str, Any]) -> Array:
#         '''Evaluate one run from a (partial) named parameter mapping.'''
#         vsem_input = make_vsem_input_from_named(par_named, driver_jnp, par_default_dict)
#         return solve_vsem_jax(vsem_input)

#     return fwd_single


# def build_vectorized_partial_forward_model(driver: Sequence[float] | Array,
#                                            par_names: Sequence[str],
#                                            par_default: Mapping[str, float] | None = None) -> Callable:
#     '''Return a vectorized forward function that accepts an array of parameter values
#     in the order `par_names` and returns model outputs for each row.

#     Example:
#        f = build_forward_model_named_vmap(driver, ['lue', 'gamma', 'veg_init'])
#        x = np.array([[0.002, 0.4, 3.0], [0.001, 0.45, 2.5]])  # shape (2,3)
#        out = f(x)  # shape (2, n_time, 5)

#     The order of par_names can be arbitrary; internally maps names to canonical positions.
#     '''
#     par_default_dict = par_default or get_vsem_default_pars_dict()
#     driver_jnp = jnp.asarray(driver)

#     # Validate names
#     for name in par_names:
#         if name not in canonical_par_names:
#             raise ValueError(f'Unknown parameter name: {name}')

#     # Create index mapping from par_names -> canonical_par_names indices
#     name_to_idx = {name: canonical_par_names.index(name) for name in par_names}

#     @jax.jit
#     def single_run_from_array(row: Array) -> Array:
#         '''Build VSEMInput from a single row of parameter values (matching par_names order).'''
#         if row.ndim != 1:
#             raise ValueError('single_from_array expects a 1D array (single run).')
#         named = {}
#         for i, name in enumerate(par_names):
#             named[name] = row[i]
#         vsem_in = make_vsem_input_from_named(named, driver_jnp, par_default_dict)
#         return solve_vsem_jax(vsem_in)

#     # Vectorize over first axis (runs)
#     vmapped = jax.vmap(single_run_from_array, in_axes=0, out_axes=0)

#     def fwd(par_array: Array) -> Array:
#         '''par_array : (n_runs, n_params) array or convertible to same'''
#         arr = jnp.asarray(par_array)
#         if arr.ndim == 1:
#             return single_run_from_array(arr)[jnp.newaxis]
#         elif arr.ndim == 2:
#             return vmapped(arr)
#         else:
#             raise ValueError('par_array must be 1D (single run) or 2D (n_runs, n_params).')

#     return fwd

# # ---------------------------------------------------------------------
# # Utilities: names getters
# # ---------------------------------------------------------------------

# def get_vsem_par_names() -> list[str]:
#     '''Return canonical parameter names (lowercase).'''
#     return list(canonical_par_names)


# def get_vsem_output_names() -> list[str]:
#     '''Return canonical output names (lowercase).'''
#     return list(canonical_output_names)


# # -----------------------------------------------------------------------------
# # Convenience parameter/driver defaults
# # -----------------------------------------------------------------------------

# def get_default_prior_bounds() -> dict[str, tuple[float, float]]:
#     '''Return uniform prior bounds for each canonical parameter (lower, upper).'''
#     return {
#         'kext': (0.2, 1.0),
#         'lar': (0.2, 3.0),
#         'lue': (5e-4, 4e-3),
#         'gamma': (0.2, 0.6),
#         'tauv': (5e2, 3e3),
#         'taus': (4e3, 5e4),
#         'taur': (5e2, 3e3),
#         'av': (2e-1, 1.0),
#         'veg_init': (0.0, 10.0),
#         'soil_init': (0.0, 30.0),
#         'root_init': (0.0, 10.0)
#     }


# def simulate_vsem_driver(key, n_days: int) -> tuple[Array, Array]:
#     '''     
#     Generates a synthetic time series at a daily time step that is supposed to
#     emulate photosynthetically active radiation (PAR) data. This time series is
#     intended for use as the model driver/forcing term of the VSEM ODE.
#     rng is a Numpy Random Generator object. 
#     '''
#     time_steps = jnp.arange(n_days)
#     driver = 10 * jnp.abs(jnp.sin(time_steps/365 * jnp.pi) + 0.25 * jr.normal(key, n_days))
#     return time_steps, driver


# # -----------------------------------------------------------------------------
# # Plotting utilities
# # -----------------------------------------------------------------------------

# def plot_vsem_outputs(output, output_names=None, nrows=1, ncols=None, figsize=(5,4), plot_kwargs=None):
#     '''
#     `output` should be (n_run, n_time_step, n_output), with outputs ordered 
#     according to the convention `get_vsem_output_names()`
#     '''

#     if output.ndim == 2:
#         output = output[np.newaxis]
#     if output_names is None:
#         output_names = get_vsem_output_names()
#     if plot_kwargs is None:
#         plot_kwargs = {}

#     n_plots = len(output_names)
#     if ncols is None:
#         ncols = int(np.ceil(n_plots / nrows))

#     fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
#     axs = np.array(axs).flatten()
#     time_steps = np.arange(output.shape[1])
#     all_output_names = get_vsem_output_names()

#     for j in range(n_plots):
#         output_name = output_names[j]
#         output_idx = all_output_names.index(output_name)

#         ax = axs[j]
#         ax.plot(time_steps, output[:,:,output_idx].T, color='gray')
#         ax.set_title(output_name)
#         ax.set_xlabel('time')
#         ax.set_ylabel(output_name)
    
#     # Hide unused axes and close figure.
#     for k in range(n_plots, nrows*ncols):
#         fig.delaxes(axs[k])

#     plt.close(fig)
#     return fig, axs