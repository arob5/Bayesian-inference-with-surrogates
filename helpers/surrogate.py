from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol
from jax.typing import ArrayLike
import jax.numpy as jnp

from inverse_problem import Distribution

Array = jnp.ndarray


class Surrogate(ABC):
    def __call__(self):
