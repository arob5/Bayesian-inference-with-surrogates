# uncprop/utils/experiment.py
"""
Utilities for running multiple replicates of surrogate-based posterior
approximation experiment.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.random as jr
from uncprop.custom_types import PRNGKey


class Replicate:
    """
    Encodes a single replicate of an experiment. Each object of the replicate
    is instantiated with a PRNG key, which is tyically the only thing that
    differentiates replicates. 
    """

    @abstractmethod
    def __init__(self, key: PRNGKey, **kwargs):
        """kwargs may be used to specify replicate settings that may vary at runtime"""
        pass

    @abstractmethod
    def __call__(self):
        """Run single replicate of experiment with specified key"""
        pass


@dataclass
class Experiment:

    name: str
    base_out_dir: Path | str
    num_reps: int
    base_key: PRNGKey
    Replicate: type[Replicate]

    def __post_init__(self):
        self.base_out_dir = Path(self.base_out_dir)
        self.out_dir = self.base_out_dir / self.name

        # create output directory
        if self.out_dir.exists():
            raise FileExistsError(f'Experiment out dir already exists: {self.out_dir}')
        print(f'Creating experiment out dir: {self.out_dir}')
        self.out_dir.mkdir(parents=True)

        # create base prng key for each experiment
        self.replicate_keys = jr.split(self.base_key, self.num_reps)

        # create replicate generator
        self.replicates = ()

    def run_replicate(self, 
                      idx: int, 
                      setup_kwargs: dict | None = None,
                      run_kwargs: dict | None = None):
        print(f'Running replicate {idx}')

        if setup_kwargs is None:
            setup_kwargs = {}
        if run_kwargs is None:
            run_kwargs = {}

        key = self.replicate_keys[idx]
        rep = self.Replicate(key=key, **setup_kwargs)
        return rep(**run_kwargs)

    def __call__(self, 
                 setup_kwargs: dict | None = None,
                 run_kwargs: dict | None = None):
        """Top-level execution of the experiment
        
        Default method iterates over each replicate and call `run_replicate()`.
        Most experiments will want to subclass and override this method to 
        handle I/O results, batching/parallelization, etc.
        """
        return [
            self.run_replicate(idx=idx,
                               setup_kwargs=setup_kwargs,
                               run_kwargs=run_kwargs)
            for idx in range(self.num_reps)                
        ]
            

class PosteriorComparison(Replicate):
    pass