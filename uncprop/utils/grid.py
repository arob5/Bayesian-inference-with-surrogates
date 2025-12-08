# uncprop/utils/grid.py
"""
Utility functions for working with discretized probability densities 
over a uniform set of grid points; i.e., the area of each cell in the 
grid is constant.

Functions use the naming convention that logp may represent (log)  
densities or point masses, while log_prob represents masses. 
logZ is used to denote (log) normalizing constants.
"""
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Sequence, Callable, Mapping

from jax.scipy.special import logsumexp
from jax.lax import cumlogsumexp

from uncprop.custom_types import Array, ArrayLike
from uncprop.core.inverse_problem import Distribution
from uncprop.utils.plot import smart_subplots

class Grid:

    def __init__(self,
                 low: ArrayLike,
                 high: ArrayLike,
                 n_points_per_dim: ArrayLike,
                 dim_names: list[str] | None):
        low = jnp.asarray(low).ravel()
        high = jnp.asarray(high).ravel()
        n_points_per_dim = jnp.asarray(n_points_per_dim, dtype=int).ravel()

        n_dims = len(low)
        if (len(high) != n_dims) or (len(n_points_per_dim) != n_dims):
            raise ValueError('low, high, and n_points_per_dim must have equal length.')
        
        if dim_names is None:
            dim_names = [f'x{i}' for i in range(n_dims)]
        else:
            if len(dim_names) != n_dims:
                raise ValueError(f'dim_names must have length equal to n_dims = f{n_dims}')

        # list of coordinates for each dimension
        coords = [
            jnp.linspace(l, h, n) for (l, h, n) in zip(low, high, n_points_per_dim)
        ]

        # two grid representations
        grid_arrays = jnp.meshgrid(*coords, indexing='xy')
        flat_grid = jnp.stack([a.ravel() for a in grid_arrays], axis=-1)

        # grid spacing
        dx = (high - low) / (n_points_per_dim - 1)

        self.low = low
        self.high = high
        self.n_points_per_dim = n_points_per_dim
        self.dx = dx
        self.dim_names = dim_names
        self.grid_arrays = grid_arrays
        self.flat_grid = flat_grid

    @property
    def cell_area(self) -> float:
        return float(jnp.prod(self.dx))
    
    @property
    def n_points(self) -> int:
        return int(jnp.prod(self.n_points_per_dim))
    
    @property
    def n_dims(self):
        return len(self.n_points_per_dim)
    
    def plot(self, 
             f: Callable | None = None,
             z: ArrayLike | None = None,
             title: str | None = None,
             ax: Axes | None = None) -> tuple[Figure, Axes]:
        """
        Visualize function values at grid points. Values must either be specified 
        directly via `z`, or a function `f` must be supplied and the values 
        obtained via `f(flat_grid)`.
        """
        
        if not ((f is None) ^ (z is None)):
            raise ValueError('Exactly one of f and z must be provided.')
        if z is None:
            z = f(self.flat_grid)

        # Ensure that z is a flat array of length equal to grid length
        z = jnp.asarray(z).ravel()
        if z.shape[0] != self.n_points:
            raise ValueError(f'Plot z values have length {z.shape[0]}; expected {self.n_points}')

        if self.n_dims == 2:
            return self._plot_2d(z, title=title, ax=ax)
        else:
            raise NotImplementedError(f'No plot() method defined for grid with n_dims = {self.n_dims}')
        
    def _plot_2d(self, z, title=None, ax=None):
        assert self.n_dims == 2

        X, Y = self.grid_arrays
        Z = z.reshape(X.shape)
        fig, ax = plot_2d_heatmap(X, Y, Z, 
                                  dim_names=self.dim_names,
                                  title=title, ax=ax)

        return fig, ax


class DensityComparisonGrid:

    def __init__(self, 
                 grid: Grid, 
                 distributions: Mapping[str, Distribution]):
        self.grid = grid
        self.distributions = distributions

        # unnormalized log densities over grid
        self.log_dens_grid = {
            nm: dist.log_density(self.grid.flat_grid).squeeze() for nm, dist in self.distributions.items()
        }

        # normalized log densities over grid
        self.log_dens_norm_grid = {
            nm: normalize_density_over_grid(logp=logp, cell_area=self.grid.cell_area)[0].squeeze() 
            for nm, logp in self.log_dens_grid.items()
        }

    def plot(self, 
             dist_names: list[str] | str, 
             normalized: bool = False, 
             log_scale: bool = True,
             ):
        
        if isinstance(dist_names, str):
            dist_names = [dist_names]

        nplots = len(dist_names)
        fig, axs = smart_subplots(nplots=nplots, max_rows=3)

        for dist_name, ax in zip(dist_names, axs):
            if normalized:
                plot_vals = self.log_dens_norm_grid[dist_name]
            else:
                plot_vals = self.log_dens_grid[dist_name]
            
            if not log_scale:
                plot_vals = jnp.exp(plot_vals)

            self.grid.plot(z=plot_vals, title=dist_name, ax=ax)

        return fig, axs


def plot_2d_heatmap(X, Y, Z, dim_names, title=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # pcolormesh expects X, Y, Z of shape (ny, nx)
    pcm = ax.pcolormesh(X, Y, Z, shading='auto')
    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    ax.set_title(title)

    return fig, ax


def normalize_density_over_grid(logp: Array, 
                                *,
                                cell_area: float | None = None, 
                                return_log: bool = True) -> tuple[Array, Array]:
    """ Normalize density over equally-spaced grid

    If `cell_area` is `None` then `logp` is interpreted as an array
    of point masses. If `cell_area` is a positive float, then this value
    is multiplied to density values to convert to masses of the grid cells.
    
    Args:
        logp: array containing log densities or masses. Can be 1d if
               representing values of single density, or 2d in which case
               each row will be normalized.
        cell_area: positive float or None, see above. Note that for 1d grids area
                   is simply length.
        return_log: If true, returns log of normalized density.

    Returns:
        tuple, containing:
          - (log) normalized version of `logp`
          - (log) normalizing constants
        Returns values on log scale if `return_log` is True. Tuple values are
        arrays of shape (n,), where n is the number of rows of logp
        (1 if logp is flat array).
    """
    logp = _check_grid_batch(logp)

    # convert from densities to masses
    if cell_area is not None:
        if not (jnp.isscalar(cell_area) and cell_area > 0):
            raise ValueError('cell_area must be a positive scalar or None')
        log_prob = logp + jnp.log(cell_area)
    else:
        log_prob = logp

    # log normalizing constant for each row
    logZ = logsumexp(log_prob, axis=1)

    # normalize
    log_prob_norm = log_prob - logZ[:,jnp.newaxis]
    logZ = logZ.ravel()

    if return_log:
        return (log_prob_norm, logZ)
    else:
        return (jnp.exp(log_prob_norm), jnp.exp(logZ))
    

def get_grid_coverage_mask(log_prob: Array, *,
                           probs: Array | None = None, 
                           check_normalized: bool = True,
                           normalized_log_tol: float = 1e-10):
    """
    Given normalized log probabilities, return indices of elements that 
    correspond to highest density regions at specified probability levels. 
    These regions are not guaranteed to be congiguous.

    Args:
        log_prob: (d,) or (n,d), representing n sets of log normalized probilities
                  over a finite set of d points.
        probs: array of probability levels. Defaults to ten evenly spaced points in [0.1, 0.99].
        check_normalized: If True, checks if each row of log_prob is normalized.
        normalized_log_tol: tolerance in log space for checking normalization.

    Returns:
        Boolean array of shape (n, n_probs, d). Element [i, j, :] gives the coverage
        mask for the ith density at probability level probs[j].
    """
    log_prob = _check_grid_batch(log_prob)

    if probs is None:
        probs = jnp.linspace(0.1, 0.99, 10)
    else:
        probs = jnp.asarray(probs).ravel()
        if jnp.any(probs < 0) or jnp.any(probs > 1):
            raise ValueError('probs must lie in [0,1]')
        
    if check_normalized:
        _check_normalized(log_prob, normalized_log_tol)

    order = jnp.argsort(log_prob, descending=True, axis=1)
    inv_order = jnp.argsort(order, axis=1)
    log_prob_sorted = log_prob[:,order]
    log_cum_prob = cumlogsumexp(log_prob_sorted, axis=1)
    
    m = probs.shape[0]
    n, d = log_prob.shape
    logp_lvls = jnp.log(probs)
    cond = (log_cum_prob[:, None, :] >= logp_lvls[None, :, None]) # shape (n, m, d)

    # For each (n, m) find smallest k s.t. cum_prob >= p). Set to -1 if never satisfied.
    any_true = jnp.any(cond, axis=2)  # shape (n, m)
    first_idx = jnp.argmax(cond, axis=2)
    first_idx = jnp.where(any_true, first_idx, -1)

    # Build sorted masks: for each k include indices 0..k (inclusive) (all False if k=1)
    idx_range = jnp.arange(d)  # shape (d,)
    mask_sorted = (idx_range[None, None, :] <= first_idx[:, :, None]) # shape (n, m, d)
    inv_idx_for_take = jnp.broadcast_to(inv_order[:, None, :], (n, m, d))
    mask = jnp.take_along_axis(mask_sorted, inv_idx_for_take, axis=2)

    return mask


def plot_2d_mask(mask: Array, 
                 grid_shape: tuple[int, int],
                 prob_idx: ArrayLike | None = None):
    if mask.ndim != 3:
        raise ValueError('mask should be 3d, as returned by `get_grid_coverage_mask()`.')
    if prob_idx is not None:
        prob_idx = jnp.atleast_1d(prob_idx)
        mask = mask[:, prob_idx, :]

    n, m, d = mask.shape
    d1, d2 = grid_shape
    if d != d1 * d2:
        raise ValueError(f'Grid shape {grid_shape} and flat mask length {d} disagree.')
    
    fig, axs = plt.subplots(nrows=n, ncols=m)

    for i in range(n):
        for j in range(m):
            ax = axs[i,j]
            mask_grid = mask[i,j,:].reshape(d1, d2).astype(int)
            ax.imshow(mask_grid, origin='lower', interpolation='nearest')
    
    return fig, axs


def coverage_curve_grid(
    logp_true: Array,
    logp_approx: Array,
    cell_area: float | None = None,
    probs: Array | None = None,
    *,
    return_masks: bool = True,
    return_weights: bool = True,
    tie_break: str = 'fractional',
    normalized_log_tol: float = 1e-10,
    rng=None
):
    """
    Compute HPD coverage curve on a discrete grid using robust log-space arithmetic.

    This function compares an *approximate* distribution (logp_approx) to a
    *reference/true* distribution (logp_true) defined on the same discrete, evenly-spaced grid.
    All inputs are provided as log-values (unnormalized log-densities). The code
    normalizes in log-space, sorts the approximate distribution to produce HPD sets, 
    and computes the mass of the *true* distribution inside those HPD sets.

    Ties (level sets) are handled robustly: by default a **fractional**
    inclusion of the last level-set is used so the approximating distribution's
    HPD mass equals the probability level exactly (avoiding systematic 
    integer-selection bias). Optionally supports other tie breaking methods
    'random' (shuffle equal values) or 'expand' (include whole level set).

    The densities are by default (`cell_area = None`) treated as log masses. If 
    `cell_area` is a positive float, then `cell_area` is multiplied by the densities
    to convert to cell masses. 

    Args:
        logp_true : array_like
            1D array of log-weights for the *true* distribution (length n = H*W)
        logp_approx : array_like
            1D array of log-weights for the *approximate* distribution (same length)
        cell_area : float or None
            Area of each grid cell. If provided it's applied as +log(cell_area) before normalization.
        probvs : array_like or None
            Array of target HPD masses (in (0,1)). If None, defaults to np.linspace(0.01,0.99,99).
        return_masks : bool
            If True, also return boolean masks (per-prob) of fully included cells.
        return_weights : bool
            If True (default), also return a 2D array `weights` shaped (len(probs), n)
            with fractional inclusion weights in [0,1] for every cell and prob. This
            encodes partial inclusion of the level-set block: 1.0 = fully included,
            0.0 = excluded, intermediate values indicate partial inclusion.
        tie_break : {'fractional','random','expand'}
            - 'fractional' (default): include fractional part of the level set so that
            the approximating mass equals the prob exactly (deterministic).
            - 'random': randomly shuffle equal-valued runs and take integer selection
            (Monte Carlo unbiased if averaged over repeats).
            - 'expand': include entire level-set (may overshoot prob).
        normalized_log_tol : float
            Tolerance in log-space for deciding whether an input is already normalized
            (i.e., |logZ| <= normalized_log_tol implies normalized).
        rng : numpy.random.Generator or None
            Optional RNG to control random tie-breaking.

    Returns:
        tuple (probs, log_coverage, masks, weights)

    Where
    - probs : array of requested probability values (same as input or default)
    - log_coverage : array of log(true_mass(HPD_approx(prob))) for each prob
    - masks : list of boolean arrays (only fully-included cells) if return_masks True. Else None.
    - weights : a float ndarray with shape (len(probs), n) with fractional weights in [0,1]
                if return_weights is True. Else None.

    Notes:
        - `masks` show which cells were fully included deterministically. For fractional
        inclusion, `weights` should be used to compute true coverage or visualization.
        - The function computes true coverage via a weighted sum over the true mass:
            true_mass = sum_i weights[prob_idx, i] * p_true[i],
        where p_true = exp(logp_true_normalized).
    """
    if probs is None:
        probs = np.linspace(0.1, 0.99, 10)
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must lie in [0,1]")

    logp_t = np.asarray(logp_true).ravel().astype(float)
    logp_a = np.asarray(logp_approx).ravel().astype(float)
    if logp_t.shape != logp_a.shape:
        raise ValueError("logp_true and logp_approx must have the same number of elements")
    n_grid = logp_t.size

    # normalize log probs, potentially converting densities to cell masses
    logp_t_norm, logZ_t = normalize_if_not(logp_t, cell_area, 
                                           tol_log=normalized_log_tol)
    logp_a_norm, logZ_a = normalize_if_not(logp_a, cell_area, 
                                           tol_log=normalized_log_tol)
    logp_t_norm = logp_t_norm.ravel()
    logp_a_norm = logp_a_norm.ravel()
    logZ_t = logZ_t.item()
    logZ_a = logZ_a.item()

    if not np.isfinite(logZ_t):
        raise ValueError("true log-probabilities are all -inf or non-finite")
    if not np.isfinite(logZ_a):
        raise ValueError("approx log-probabilities are all -inf or non-finite")

    # For weighted sums
    p_t = np.exp(logp_t_norm)

    # sorted descending approximate log-probs
    # for reproducibility with 'random' tie-break, optionally shuffle equal-values
    order = np.argsort(logp_a_norm)[::-1]
    logp_a_sorted = logp_a_norm[order]


    log_cum = logcumsumexp_1d(logp_a_sorted)  # log of cumulative approx mass in sorted order

    log_coverage = np.empty_like(probs, dtype=float)
    masks = [] if return_masks else None
    weights = np.zeros((probs.size, n_grid), dtype=float) if return_weights else None

    for prob_idx, prob in enumerate(probs):
        if prob <= 0:
            log_coverage[prob_idx] = -np.inf
            if return_masks:
                masks.append(np.zeros(n_grid, dtype=bool))
            if return_weights:
                weights[prob_idx, :] = 0.0
            continue

        if prob >= 1:
            mask_full = np.ones(n_grid, dtype=bool)
            log_cov_full = logsumexp(logp_t_norm)  # should be 0 if normalized
            log_coverage[prob_idx] = log_cov_full
            if return_masks:
                masks.append(mask_full)
            if return_weights:
                weights[prob_idx, :] = 1.0
            continue

        log_prob = np.log(prob)

        # First index where cumulative >= prob
        k = None
        exceeds_prob = (log_cum >= log_prob) & np.isfinite(log_cum)
        if np.any(exceeds_prob):
            k = np.argmax(exceeds_prob)  # first True index
        else:
            k = None

        # shouldn't happen since should sum to one, but possible
        # due to numerical error; fallback to include all
        if k is None:
            mask_full = np.ones(n_grid, dtype=bool)
            log_cov = logsumexp(logp_t_norm[mask_full])
            log_coverage[prob_idx] = log_cov
            if return_masks:
                masks.append(mask_full)
            if return_weights:
                weights[prob_idx, :] = 1.0
            continue

        # find contiguous block of entries equal to threshold
        # (contiguous due to fact that array is sorted)
        threshold = logp_a_sorted[k]
        approx_equal = np.isclose(logp_a_sorted, threshold, rtol=0, atol=1e-12)
        left_idx = k
        while left_idx > 0 and approx_equal[left_idx - 1]:
            left_idx -= 1
        right_idx = k
        while right_idx + 1 < n_grid and approx_equal[right_idx + 1]:
            right_idx += 1

        if tie_break == 'expand':
            sel_sorted_indices = np.arange(0, right_idx + 1)
            mask = np.zeros(n_grid, dtype=bool)
            mask[order[sel_sorted_indices]] = True
            if return_weights:
                weights[prob_idx, :] = 0.0
                weights[prob_idx, order[: (right_idx + 1)]] = 1.0
            log_cov = logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        elif tie_break == 'random':
            sel_sorted_indices = np.arange(0, k + 1)
            mask = np.zeros(n_grid, dtype=bool)
            mask[order[sel_sorted_indices]] = True
            if return_weights:
                weights[prob_idx, :] = 0.0
                weights[prob_idx, order[: (k + 1)]] = 1.0
            log_cov = logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        else:  # tie_break == 'fractional'
            # compute cumulative mass before the block
            if left_idx == 0:
                cum_before = 0.0
            else:
                log_cum_before = log_cum[left_idx - 1]
                cum_before = float(np.exp(log_cum_before))

            block_log_mass = logsumexp(logp_a_sorted[left_idx: right_idx + 1])
            block_mass = float(np.exp(block_log_mass))

            if block_mass <= 0:
                f = 0.0
            else:
                f = float((prob - cum_before) / block_mass)
                f = np.clip(f, 0.0, 1.0)

            # build weights: 1 for fully included before the block, f for block, 0 otherwise
            if return_weights:
                w = np.zeros(n_grid, dtype=float)
                if left_idx > 0:
                    w[order[:left_idx]] = 1.0
                w[order[left_idx: right_idx + 1]] = f
                weights[prob_idx, :] = w

            # compute true mass via weighted sum in linear space for stability
            if return_weights:
                total_true = float(np.dot(weights[prob_idx, :], p_t))
            else:
                # if weights not requested, fall back to computing true_before and true_block
                if left_idx > 0:
                    idx_before = order[:left_idx]
                    true_before = float(np.sum(p_t[idx_before]))
                else:
                    true_before = 0.0
                idx_block = order[left_idx: right_idx + 1]
                true_block = float(np.sum(p_t[idx_block]))
                total_true = true_before + f * true_block

            if total_true <= 0:
                log_cov = -np.inf
            else:
                log_cov = np.log(total_true)

            # build boolean mask of fully-included cells (before block)
            mask = np.zeros(n_grid, dtype=bool)
            if left_idx > 0:
                mask[order[:left_idx]] = True

        log_coverage[prob_idx] = log_cov
        if return_masks:
            masks.append(mask.copy())

    return probs, log_coverage, masks, weights


def _is_normalized(log_prob: Array, log_tol: float = 1e-10) -> Array:
    """
    Check if density is normalized, given log probabilities. Vectorized
    to operate over rows of `log_prob`.
    """
    log_prob = _check_grid_batch(log_prob)
    logZ = logsumexp(log_prob, axis=1)

    n = log_prob.shape[1]
    is_normalized = jnp.tile(False, n)
    is_finite = jnp.isfinite(logZ)
    is_normalized = is_normalized.at[is_finite].set(
        jnp.less_equal(jnp.abs(logZ[is_finite]), log_tol)
    )   

    return is_normalized


def _check_normalized(log_prob: Array, log_tol: float = 1e-10) -> None:
    is_normalized = jnp.all(_is_normalized(log_prob, log_tol))
    if not is_normalized:
        raise ValueError('log_prob is not normalized.')


def _check_grid_batch(x: ArrayLike) -> Array:
    """ 
    Converts to shape (n,d), where each row represents a discretized 
    density over the same grid points. Converts (d,) input to (1,d).
    """
    x = jnp.asarray(x)
    if x.ndim > 2:
        raise ValueError('Invalid input: x.ndim > 2')
    
    return jnp.atleast_2d(x)