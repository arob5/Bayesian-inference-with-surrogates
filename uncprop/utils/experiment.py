# uncprop/utils/experiment.py
"""
Utilities for running multiple replicates of surrogate-based posterior
approximation experiment.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
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
    def __call__(self, *args, **kwargs) -> Any:
        """Run single replicate of experiment with specified key"""
        pass


@dataclass
class Experiment:
    """
    Run an experiment consisting of a set of replicates of a base experiment.

    Instantiating an Experiment object stores the Replicate that will be used, 
    as well as additional metadata. It also handles filepaths and creates
    the JAX PRNG keys used for each replicate.

    `base_out_dir` is the base directory for the experiment. Each time 
    `__call__()` is invoked, a new subdirectory is created, which houses the
    output from that particular call. This allows running an experiment with 
    a set structure/PRNG keys with different settings.
    """
    name: str
    base_out_dir: Path | str
    num_reps: int
    base_key: PRNGKey
    Replicate: type[Replicate]
    subdir_name_fn: Callable[[dict, dict], str | Path]
    write_to_file: bool = True

    def __post_init__(self):
        self.base_out_dir = Path(self.base_out_dir)

        # create output directory
        if self.write_to_file: 
            if self.base_out_dir.exists():
                print(f'Using existing base output directory: {self.base_out_dir}')
            else:
                print(f'Creating new output directory: {self.base_out_dir}')
                self.base_out_dir.mkdir(parents=True)

        # create base prng key for each replicate
        self.replicate_keys = jr.split(self.base_key, self.num_reps)

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
    
    def save_results(self, subdir: Path, *args, **kwargs):
        print('No save_results() methods implemented.')

    def make_subdir_name(self, setup_kwargs: dict, run_kwargs: dict) -> str | Path:
        return self.subdir_name_fn(setup_kwargs, run_kwargs)

    def create_subdir(self, 
                      *,
                      setup_kwargs: dict, 
                      run_kwargs: dict, 
                      name: str | Path | None = None):
        if not self.write_to_file:
            return None

        if name is None:
            subdir = self.make_subdir_name(setup_kwargs, run_kwargs)
        else: 
            subdir = Path(name)

        subdir = self.base_out_dir / subdir

        # create output directory
        if subdir.exists():
            raise FileExistsError(f'Experiment sub-directory already exists: {subdir}')
        print(f'Creating experiment sub-directory: {subdir}')
        subdir.mkdir(parents=True)

        return subdir

    def __call__(self,
                 setup_kwargs: dict | None = None,
                 run_kwargs: dict | None = None,
                 backup_frequency: int | None = None,
                 subdir_name: str | Path | None = None):
        """Top-level execution of the experiment
        
        Default method iterates over each replicate and call `run_replicate()`, 
        optionally saving results periodically by calling `save_results()`.
        Some experiments may want to subclass and override to handle specialty
        batching/parallelization.

        `subdir` allows manually specifying the subdirectory for this call,
        overriding the default behavior of using `make_subdir_name()`.
        """
        subdir = self.create_subdir(setup_kwargs=setup_kwargs, 
                                    run_kwargs=run_kwargs,
                                    name=subdir_name)

        results: list[Any] = []
        failed_reps: list[int] = []

        for rep_idx in range(self.num_reps):
            try:
                rep_result = self.run_replicate(idx=rep_idx,
                                                setup_kwargs=setup_kwargs,
                                                run_kwargs=run_kwargs)
                results.append(rep_result)
            except Exception as e:
                print(f'Iteration {rep_idx} failed with error: {e}')
                failed_reps.append(rep_idx)
                results.append(e)
            finally:
                final_iteration = (rep_idx == self.num_reps-1)
                backup_iteration = ((backup_frequency is not None) and 
                                    (rep_idx % backup_frequency == 0) and 
                                    (rep_idx != 0))               

                if self.write_to_file and (final_iteration or backup_iteration):
                    try:
                        self.save_results(subdir, results, failed_reps)
                    except Exception as save_error:
                        print(f'save_results() failed with error: {save_error}')

        print(f'{len(failed_reps)} of {self.num_reps} replicates failed.')
        return results, failed_reps
