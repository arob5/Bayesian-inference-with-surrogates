# experiments/vsem/grid_utils.py
"""
Utilities for normalizing densities and computing coverage metrics over 2d grid.
"""
from __future__ import annotations

import numpy as np

def _logsumexp(vec):
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return -np.inf
    m = np.max(vec)
    if not np.isfinite(m):
        return -np.inf
    s = np.sum(np.exp(vec - m))
    return m + np.log(s)

def _logcumsumexp_1d(logx):
    logx = np.asarray(logx, dtype=float)
    n = logx.size
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    out[0] = logx[0]
    for i in range(1, n):
        a = out[i-1]
        b = logx[i]
        if a == -np.inf and b == -np.inf:
            out[i] = -np.inf
        else:
            m = a if a > b else b
            out[i] = m + np.log(np.exp(a - m) + np.exp(b - m))
    return out


def log_probs_are_normalized(logp, tol_log: float = 1e-10) -> tuple[bool, float]:
    """ Check whether density is (approximately) normalized.

    Given an array of log density values, test whether the exponentiated values
    are normalized; i.e., whether they sum to one. Note that this treats the 
    values as point masses - no correction is made for grid cell size.

    Returns:
        tuple, with values:
            is_normalized: bool 
            logZ: the log of the sum of the density values
    """
    logp = np.asarray(logp, dtype=float)
    logZ = _logsumexp(logp)
    if not np.isfinite(logZ):
        return False, logZ
    
    is_normalized = (abs(logZ) <= tol_log)
    return is_normalized, logZ


def normalize_over_grid(log_dens, cell_area: float | None = None, *, 
                        return_log: bool = True):
    """ Normalize density over equally-spaced 2d grid

    If `cell_area` is `None` then `log_dens` is interpreted as an array
    of point masses. If `cell_area` is a positive float, then this value
    is multiplied to density values to convert to masses of the grid cells.
    
    Args:
        log_dens: array containing log densities or masses. Can be 1d if
            representing values of single density, or 2d in which case
            each row will be normalized.

    Returns:
        tuple, containing:
          - (log) normalized version of `log_dens`
          - (log) normalizing constants
        Returns values on log scale if `return_log` is True. Tuple values are
        floats if `log_dens` is 1d, else they are arrays.
    """
    log_dens = np.asarray(log_dens, dtype=float)
    single = (log_dens.ndim == 1)
    log2d = np.atleast_2d(log_dens)

    # convert from densities to masses
    if cell_area is not None:
        if not (np.isscalar(cell_area) and cell_area > 0):
            raise ValueError("cell_area must be a positive scalar or None")
        lp = log2d + np.log(cell_area)
    else:
        lp = log2d

    # row-wise logZ
    logZ_list = np.array([_logsumexp(row) for row in lp])

    with np.errstate(invalid='ignore'): # may be all -Inf
        log_norm = lp - logZ_list[:, np.newaxis]
    if single:
        if return_log:
            return (log_norm.ravel(), logZ_list.item())
        else:
            return np.exp(log_norm.ravel()), np.exp(logZ_list.item())
    else:
        if return_log:
            return (log_norm, logZ_list)
        else:
            return (np.exp(log_norm), np.exp(logZ_list))


def coverage_curve_grid(
    logp_true,
    logp_approx,
    cell_area=None,
    probs=None,
    *,
    return_masks=True,
    return_weights=True,
    tie_break='fractional',
    normalized_log_tol=1e-10,
    rng=None
):
    """ Compute coverage curve over 2d grid with robust log-space normalization and tie handling.

    tie_break:
      - 'fractional' : if threshold falls in a level set, include a fractional portion
                       of that level set so the approx mass equals prob exactly.
      - 'random'     : break ties by random shuffling of equal-valued cells.
      - 'expand'     : include the whole level set.

    Returns:
        tuple, containing:
            - probs: the `probs` argument containing the array of probability levels
            - log_coverage
            - calib_error_log
            - [masks]: 
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
    is_norm_t, logZ_t_guess = log_probs_are_normalized(logp_t, tol_log=normalized_log_tol)
    if is_norm_t:
        logp_t_norm = logp_t.copy()
        logZ_t = 0.0
    else:
        logp_t_norm, logZ_t = normalize_over_grid(logp_t, cell_area, return_log=True)

    is_norm_a, logZ_a_guess = log_probs_are_normalized(logp_a, tol_log=normalized_log_tol)
    if is_norm_a:
        logp_a_norm = logp_a.copy()
        logZ_a = 0.0
    else:
        logp_a_norm, logZ_a = normalize_over_grid(logp_a, cell_area, return_log=True)

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

    if tie_break == 'random':
        # shuffle equal-value runs to randomize ties
        if rng is None:
            rng = np.random.default_rng()
        # find runs of equal values (within isclose)
        i = 0
        while i < n_grid:
            thr = logp_a_sorted[i]
            # find run end where values close to thr
            j = i + 1
            while j < n_grid and np.isclose(logp_a_sorted[j], thr, atol=0, rtol=1e-12):
                j += 1
            if j - i > 1:
                # random permute indices in this run
                run_idx = np.arange(i, j)
                perm = rng.permutation(run_idx)
                logp_a_sorted[run_idx] = logp_a_sorted[perm]
                order[run_idx] = order[perm]
            i = j

    log_cum = _logcumsumexp_1d(logp_a_sorted)  # log of cumulative approx mass in sorted order

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
            mask_full = np.ones(n, dtype=bool)
            log_cov_full = _logsumexp(logp_t_norm)  # should be 0 if normalized
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
            log_cov = _logsumexp(logp_t_norm[mask_full])
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
            log_cov = _logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        elif tie_break == 'random':
            sel_sorted_indices = np.arange(0, k + 1)
            mask = np.zeros(n_grid, dtype=bool)
            mask[order[sel_sorted_indices]] = True
            if return_weights:
                weights[prob_idx, :] = 0.0
                weights[prob_idx, order[: (k + 1)]] = 1.0
            log_cov = _logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        else:  # tie_break == 'fractional'
            # compute cumulative mass before the block
            if left_idx == 0:
                cum_before = 0.0
            else:
                log_cum_before = log_cum[left_idx - 1]
                cum_before = float(np.exp(log_cum_before))

            block_log_mass = _logsumexp(logp_a_sorted[left_idx: right_idx + 1])
            block_mass = float(np.exp(block_log_mass))

            if block_mass <= 0:
                f = 0.0
            else:
                f = float((prob - cum_before) / block_mass)
                f = max(0.0, min(1.0, f))

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
            mask = np.zeros(n, dtype=bool)
            if left > 0:
                mask[order[:left]] = True

        log_coverage[prob_idx] = log_cov
        calib_error_log[prob_idx] = (log_cov - np.log(prob)) if np.isfinite(log_cov) else np.nan
        if return_masks:
            masks.append(mask.copy())

    if return_masks:
        return probs, log_coverage, masks
    return probs, log_coverage
