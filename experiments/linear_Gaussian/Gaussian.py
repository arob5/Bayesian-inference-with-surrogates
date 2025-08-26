# Gaussian.py
from __future__ import annotations

import math
import copy
import numpy as np
from scipy.linalg.blas import dtrmm
from scipy.linalg import cholesky, qr, solve_triangular

# Constants.
LOG_TWO_PI = math.log(2.0 * math.pi)

# Helper functions.
def mult_A_L(A, L):
    return dtrmm(side=1, a=L, b=A, alpha=1.0, trans_a=0, lower=1)

def mult_A_Lt(A, L):
    return dtrmm(side=1, a=L, b=A, alpha=1.0, trans_a=1, lower=1)


class Gaussian:
    def __init__(self,
                 mean: np.ndarray|None = None,
                 cov: np.ndarray|None = None,
                 chol: np.ndarray|None = None,
                 store: str = "chol",
                 rng: np.random.Generator|None = None) -> None:
        """
        `store` can be "chol", "cov", or "both".
        """

        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # Defaults to standard Gaussian when insufficient info is provided.
        if all(x is None for x in [mean, cov, chol]):
            self.mean = np.zeros(1)
            cov = np.ones((1,1))
        elif mean is None:
            dim = cov.shape[0] if cov is not None else chol.shape[0]
            self.mean = np.zeros(dim)
        elif cov is None and chol is None:
            dim = mean.shape[0]
            chol = np.eye(dim)
            self.mean = mean
        else:
            self.mean = mean

        self.set_cov_info(cov, chol, store)
        # self._ensure_consistent_dims() # TODO: write this method.

    @property
    def cov(self) -> np.ndarray:
        if self._cov is not None:
            return self._cov
        return self._chol @ self._chol.T

    @property
    def chol(self) -> np.ndarray:
        if self._chol is not None:
            return self._chol
        return cholesky(self._cov, lower=True)

    def set_cov_info(self, cov: np.ndarray = None, chol: np.ndarray = None,
                     store: str = "chol") -> None:

        if store not in {"chol", "cov", "both"}:
            raise ValueError("`store` must be either 'chol', 'cov', or 'both'.")

        if cov is None and chol is None:
            raise ValueError("Gaussian._set_cov_info() requires either ",
                             "`cov` or `chol` to be provided.")

        # Store Cholesky factor.
        if (store == "chol") or (store == "both"):
            if chol is not None:
                self._chol = chol
            else:
                self._chol = cholesky(cov, lower=True)

        # Store covariance matrix.
        if (store == "cov") or (store == "both"):
            if cov is not None:
                self._cov = cov
            else:
                self._cov = chol @ chol.T

        # If not storing both, delete any existing values to avoid misalignment.
        if store == "cov":
            self._chol = None
        elif store == "chol":
            self._cov = None

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, num_samp: int = 1) -> np.ndarray:
        """
        Returns (num_samp, dim) array containing `num_samp` iid samples from
        the Gaussian stacked in the rows of the array.
        """
        Z = self.rng.normal(size=(num_samp, self.dim))
        return self.mean + mult_A_Lt(Z, self.chol)

    def log_p(self, x: np.ndarray) -> np.ndarray:
        """
        `x` must be an (n,d) or (d,) array. The returned array is either
        (n,) or (1,), respectively.
        """

        if x.ndim == 1:
            x = x.reshape(1, -1)

        # solve L z = y to obtain z = L^{-1} y, where y=x-mean.
        L = self.chol
        z = solve_triangular(L, (x-self.mean).T, lower=True, check_finite=False)

        # Quadratic term.
        quad_term = np.sum(z * z, axis=0)

        # Log determinant term.
        logdet_term = 2 * np.sum(np.log(np.diag(L)))

        return -0.5 * (quad_term + logdet_term + self.dim * LOG_TWO_PI)

    def apply_affine_map(self, A: np.ndarray|None = None, b: np.ndarray|None = None,
                         store: str = "chol") -> Gaussian:
        """
        Returns the Gaussian resulting by sending the current Gaussian (self)
        through an affine map F(x) = Ax + b. If x ~ N(m, C) then:
        y := F(x) ~ N(Am + b, ACA^T).
        """
        # If no affine map is specified, defaults to identity map.
        if A is None and b is None:
            return copy.deepcopy(self)
        elif A is None: # Just shift mean.
            y = copy.deepcopy(self)
            y.mean += b
            return y
        elif b is None:
            b = np.zeros(A.shape[0])

        # Computing Cholesky of ACA^T = (AL)(AL)^T. AL is a sqrt; turn into
        # Cholesky factor via QR decomposition.
        S = mult_A_L(A, self.chol)
        if store == "chol":
            Q, R = qr(S.T, mode="economic")
            chol = R.T
            cov = None
        else:
            cov = S @ S.T
            chol = None

        return Gaussian(mean= A @ self.mean + b,
                        cov=cov, chol=chol, store=store)


    def convolve_with_Gaussian(self, A: np.ndarray|None = None, b: np.ndarray|None = None,
                               cov_new: np.ndarray|None = None, chol_new: np.ndarray|None = None,
                               store: str = "chol") -> Gaussian:
        """
        Returns the Gaussian resulting from convolving the current Gaussian (self)
        with a new Gaussian. Let x ~ N(m, C) denote the new Gaussian. The
        method returns:
        E_x[N(. | Ax + b, C_new)] = N(. | Am + b, ACA^T + C_new).

        The covariance C_new is specified either via `cov_new` or `chol_new`
        (`cov_new` takes precedence).
        """
        # Intermediate Gaussian.
        y = self.apply_affine_map(A, b, store)

        # Default to identity covariance if not provided.
        if cov_new is None and chol_new is None:
            cov_new = np.eye(y.dim)
        elif cov_new is None:
            cov_new = chol_new @ chol_new.T

        y.set_cov_info(cov= y.cov + cov_new, store=store)
        return y

    def invert_affine_Gaussian(self, y: np.ndarray, A: np.ndarray,
                               b: np.ndarray|None = None, cov_noise: np.ndarray|None = None,
                               chol_noise: np.ndarray|None = None,
                               store: str = "chol") -> Gaussian:
        # TODO: currently performs "data space" update. Should update this
        # to choose whether to perform "data space" vs. "parameter space"
        # update based on dims and which Cholesky factors are provided.

        dim_out = y.shape[0]
        if A.shape[0] != dim_out:
            raise ValueError("`invert_affine_Gaussian()` dimension mismatch ",
                             "between `A` and `y`.")

        # Default to zero shift in the affine map, and identity noise covariance.
        if b is None:
            b = np.zeros(dim_out)
        if cov_noise is None and chol_noise is None:
            cov_noise = np.eye(dim_out)
        if cov_noise is None:
            cov_noise = chol_noise @ chol_noise.T

        # Chokesky factorize ACA^T + cov_noise
        L_prior = self.chol
        B = L_prior @ (A @ L_prior).T
        L_post = cholesky(A @ B + cov_noise, lower=True)

        # Posterior mean.
        r = y - b - A @ self.mean
        v = solve_triangular(L_post.T, solve_triangular(L_post, r, lower=True), lower=False)
        mean_post = self.mean + B @ v

        # Posterior covariance.
        C = solve_triangular(L_post, B.T, lower=True)
        cov_post = self.cov - C @ C.T

        return Gaussian(mean=mean_post, cov=cov_post, store=store)
