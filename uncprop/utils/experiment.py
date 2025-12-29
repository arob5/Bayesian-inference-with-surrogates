# uncprop/utils/experiment.py
"""
Utilities for running multiple replicates of surrogate-based posterior
approximation experiment.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import numpy as np
import jax.numpy as jnp
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
        self.base_out_dir: Path = Path(self.base_out_dir)

        # create output directory 
        if self.base_out_dir.exists():
            self._validate_existing_experiment()
            print(f'Continuing existing experiment: {self.base_out_dir}')
        else:
            self._init_new_experiment()

        # generate keys for each replicate
        self.replicate_keys = jr.split(self.base_key, self.num_reps)


    def _validate_existing_experiment(self):
        """
        Load an existing experiment with base directory `base_out_dir`.
        Ensures that the current base PRNG key matches the key data saved
        for the existing experiment.
        """
        base_key_data_path = self.base_out_dir / 'base_key.npy'

        if not base_key_data_path.exists():
            raise FileNotFoundError(f'Existing experiment missing key data: {base_key_data_path}')
        
        key_data = jnp.load(base_key_data_path)
        base_key = jr.wrap_key_data(key_data)

        if base_key != self.base_key:
            raise ValueError('Existing experiment base key does not match current base key.')


    def _init_new_experiment(self):
        print(f'Creating new experiment: {self.base_out_dir}')
        self.base_out_dir.mkdir(parents=True)

        # save key data
        jnp.save(self.base_out_dir / 'base_key.npy', jr.key_data(self.base_key))


    def init_replicate(self, 
                       rep_idx: int,
                       setup_kwargs: dict,
                       rep_subdir: Path):
        print(f'Initializing replicate {rep_idx}')
        key = self.replicate_keys[rep_idx]

        start = time.perf_counter()
        rep = self.Replicate(key=key, out_dir=rep_subdir, **setup_kwargs)
        end = time.perf_counter()
        print(f'\tSetup time: {end - start:.6f} seconds')

        return rep
        

    def run_replicate(self,
                      rep: Replicate, 
                      rep_idx: int,
                      run_kwargs: dict,
                      rep_subdir: Path):
        print(f'Running replicate {rep_idx}')
        key = self.replicate_keys[rep_idx]
        _, key_run = jr.split(key)

        start = time.perf_counter()
        rep_results = rep(key=key_run, out_dir=rep_subdir, **run_kwargs)
        end = time.perf_counter()
        print(f'\tRun time: {end - start:.6f} seconds')

        return rep_results


    def make_subdir_name(self, setup_kwargs: dict, run_kwargs: dict) -> str | Path:
        """Create subdirectory name as function of {setup_kwargs, run_kwargs}"""
        return self.subdir_name_fn(setup_kwargs, run_kwargs)

    def create_subdir(self, 
                      *,
                      setup_kwargs: dict, 
                      run_kwargs: dict, 
                      name: str | Path | None = None):
        """Create subdirectory for current experiment call"""

        if name is None:
            subdir = self.make_subdir_name(setup_kwargs, run_kwargs)
        else: 
            subdir = Path(name)

        subdir = self.base_out_dir / subdir

        # create output directory
        if subdir.exists():
            print(f'Working in existing experiment sub-directory: {subdir}')
        else:
            print(f'Creating experiment sub-directory: {subdir}')        
            subdir.mkdir(parents=True)

        return subdir


    def __call__(self,
                 rep_idx: Sequence[int] | int | None = None,                 
                 setup_kwargs: dict | None = None,
                 run_kwargs: dict | None = None,
                 subdir_name: str | Path | None = None,
                 write_to_log_file: bool = True,
                 overwrite: bool = False):
        """ Top-level execution of the experiment

        Runs the replicates indexed by `rep_idx`, with the default None
        running all replicates. The output subdirectory for this call is
        determined by {setup_kwargs, run_kwargs, subdir_name}
        (see `create_subdir()` method).

        Replicates are run sequentially in a loop. Exceptions are caught
        so that execution is not stopped if a replicate fails. 
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # replicates to run
        if isinstance(rep_idx, int):
            rep_idx = [rep_idx]
        elif rep_idx is None:
            rep_idx = list(range(self.num_reps))

        setup_kwargs = setup_kwargs or {}
        run_kwargs = run_kwargs or {}

        # subdirectory name for outputs
        subdir = self.create_subdir(setup_kwargs=setup_kwargs, 
                                    run_kwargs=run_kwargs,
                                    name=subdir_name)
        subdir_logs = subdir / 'log'
        if not subdir_logs.exists():
            subdir_logs.mkdir()
        
        experiment_run_kwargs = {'rep_idx': rep_idx, 
                                 'setup_kwargs': setup_kwargs,
                                 'run_kwargs': run_kwargs,
                                 'subdir': subdir,
                                 'overwrite': overwrite}

        if write_to_log_file:
            logfile_path = subdir_logs / f'experiment_call_{timestamp}.log'
            
            with open(logfile_path, 'w') as log_file:
                with redirect_stdout(log_file):
                    results, failed_reps, skipped_reps = self._run_experiment(**experiment_run_kwargs)
        else:
            results, failed_reps, skipped_reps = self._run_experiment(**experiment_run_kwargs)

        np.savetxt(subdir_logs / f'failed_reps_{timestamp}.txt', failed_reps, fmt='%d')
        np.savetxt(subdir_logs / f'skipped_reps_{timestamp}.txt', skipped_reps, fmt='%d')
        return results, failed_reps, skipped_reps


    def _run_experiment(self,
                        rep_idx: Sequence[int],                 
                        setup_kwargs: dict,
                        run_kwargs: dict,
                        subdir: Path,
                        overwrite: bool):
        """Lower-level function for experiment execution

        Intended to be called by __call__().
        """

        print('Experiment call info:')
        print(f'\toutput directory: {subdir}')
        print(f'\toverwrite existing output: {overwrite}')
        print(f'\trep indices: {rep_idx}')
        
        results: list[Any] = []
        failed_reps: list[int] = []
        skipped_reps: list[int] = []

        for idx in rep_idx:
            try:
                rep_subdir, skip = self._make_and_validate_rep_subdir(subdir, idx, overwrite)
                if skip:
                    print(f'Skipping iteration {idx}')
                    skipped_reps.append(idx)
                    continue

                rep = self.init_replicate(idx, setup_kwargs, rep_subdir)
                rep_result = self.run_replicate(rep, idx, run_kwargs, rep_subdir)
                results.append(rep_result)
            except Exception as e:
                print(f'Iteration {idx} failed with error:')
                print(e)
                failed_reps.append(idx)
                results.append(e)

        print(f'{len(failed_reps)} of {self.num_reps} replicates failed.')
        print(f'{len(skipped_reps)} of {self.num_reps} replicates skipped.')

        return results, failed_reps, skipped_reps
    

    def _make_and_validate_rep_subdir(self, subdir: Path, rep_idx: int, overwrite: bool):
        """
        If rep subdir does not exist, creates it and saves rep key to file.
        If it already exists and overwrite is False, returns bool indicating 
        rep should be skipped. If it already exists and overwrite is True, 
        ensures that the rep key agrees with the current key.
        """

        rep_subdir = subdir / f'rep{rep_idx}'

        if not rep_subdir.exists():
            rep_subdir.mkdir()

        rep_key_path = rep_subdir / 'rep_key.npy'

        # TODO: temp hack for PDE experiment
        # if not overwrite and rep_key_path.exists():
        #     return rep_subdir, True

        samp_path = rep_subdir / 'samples.npz'
        if not overwrite and samp_path.exists():
            return rep_subdir, True
        ###
        
        if rep_key_path.exists():
            key_data = jnp.load(rep_key_path)
            rep_key = jr.wrap_key_data(key_data)
            if rep_key != self.replicate_keys[rep_idx]:
                raise ValueError(
                    f'rep key for rep {rep_idx} does not match existing key'
                    f'Either the underlying code changed or iteration numbers were mixed up.'
                )
            print(f'Overwriting results for rep {rep_idx}')
        else:
            jnp.save(rep_key_path, jr.key_data(self.replicate_keys[rep_idx]))

        return rep_subdir, False