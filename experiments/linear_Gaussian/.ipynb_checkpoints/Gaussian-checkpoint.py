# Gaussian.py

import math
import numpy as np
from scipy.linalg import solve_triangular

# Constants.
LOG_TWO_PI = math.log(2.0 * math.pi)

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

        self._set_cov_info(cov, chol, store)
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
        return np.linalg.cholesky(self._cov, upper=False)

    @cov.setter
    def cov(self, cov: np.ndarray, store: str = "chol") -> None:
        self._set_cov_info(cov=cov, store=store)

    @chol.setter
    def chol(self, chol: np.ndarray, store: str = None) -> None:
        self._set_cov_info(chol=chol, store=store)

    def _set_cov_info(self, cov: np.ndarray = None, chol: np.ndarray = None,
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
                self._chol = np.linalg.cholesky(cov, upper=False)

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
        return self.mean + self.rng.normal(size=(num_samp, self.dim)) @ self.chol

    def log_p(self, x: np.ndarray) -> np.ndarray:
        """
        `x` must be an (n,d) or (d,) array. The returned array is either
        (n,) or (1,), respectively.
        """

        if x.ndim == 1:
            x = x.reshape(1, -1)

        # solve L z = y to obtain z = L^{-1} y, where y=x-mean.
        z = solve_triangular(self.chol, (x-self.mean).T, lower=True, check_finite=False)

        # Quadratic term.
        quad_term = np.sum(z * z, axis=0)

        # Log determinant term.
        logdet_term = 2 * np.sum(np.log(np.diag(L)))

        return -0.5 * (quad_term + logdet_term + d*LOG_TWO_PI)
