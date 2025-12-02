# uncprop/utils/grid.py
"""
Utility functions for working with uniform grids of points; i.e., the 
area of each cell in the grid is constant.
"""
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Sequence

from jax.scipy.special import logsumexp
from jax.lax import cumlogsumexp


Array = jnp.ndarray


def normalize_density_over_grid(log_p: Array, 
                                *,
                                cell_area: float | None = None, 
                                return_log: bool = True) -> tuple[Array, Array]:
    """ Normalize density over equally-spaced grid

    If `cell_area` is `None` then `log_p` is interpreted as an array
    of point masses. If `cell_area` is a positive float, then this value
    is multiplied to density values to convert to masses of the grid cells.
    
    Args:
        log_p: array containing log densities or masses. Can be 1d if
               representing values of single density, or 2d in which case
               each row will be normalized.
        cell_area: positive float or None, see above. Note that for 1d grids area
                   is simply length.
        return_log: If true, returns log of normalized density.

    Returns:
        tuple, containing:
          - (log) normalized version of `log_p`
          - (log) normalizing constants
        Returns values on log scale if `return_log` is True. Tuple values are
        arrays of shape (n,), where n is the number of rows of log_p
        (1 if log_p is flat array).
    """
    log_p = jnp.asarray(log_p)
    if log_p.ndim > 2:
        raise ValueError('normalize_over_grid() requires log_p.ndim <= 2')
    log_p = jnp.atleast_2d(log_p)

    # convert from densities to masses
    if cell_area is not None:
        if not (jnp.isscalar(cell_area) and cell_area > 0):
            raise ValueError('cell_area must be a positive scalar or None')
        log_prob = log_p + jnp.log(cell_area)
    else:
        log_prob = log_p

    # log normalizing constant for each row
    logZ = logsumexp(log_prob, axis=1)

    # normalize
    log_prob_norm = log_prob - logZ[:,jnp.newaxis]
    logZ = logZ.ravel()

    if return_log:
        return (log_prob_norm, logZ)
    else:
        return (jnp.exp(log_prob_norm), jnp.exp(logZ))