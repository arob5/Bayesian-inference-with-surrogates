"""
Minimal viable test for the VSEM experiment pipeline.

Runs a single replicate with reduced settings (small grid, few design points,
short MCMC chains) to verify the experiment code works end-to-end.
This does NOT run the full paper experiment — see experiments/vsem/runner.py for that.
"""
from jax import config
config.update('jax_enable_x64', True)

import importlib.util
import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

# Import experiment.py by path to avoid module name collisions
REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "vsem_experiment", REPO_ROOT / "experiments" / "vsem" / "experiment.py"
)
_vsem_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vsem_exp)
VSEMReplicate = _vsem_exp.VSEMReplicate

from uncprop.utils.experiment import Experiment


def test_vsem_replicate_setup():
    """
    Test that a single VSEM replicate can be set up: generates synthetic data,
    fits the surrogate GP, and computes predictions on a grid. This exercises
    the core model (vsemjax), inverse problem setup, and surrogate fitting.
    """
    key = jr.key(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        rep = VSEMReplicate(
            key=key,
            out_dir=out_dir,
            n_design=4,
            surrogate_tag='gp',
            n_grid=5,
            surrogate_settings={'verbose': False, 'jitter': 1e-4},
        )

        # Verify core components were created
        assert rep.posterior is not None, "Posterior not created"
        assert rep.posterior_surrogate is not None, "Surrogate posterior not created"
        assert rep.grid is not None, "Grid not created"
        assert rep.surrogate_pred is not None, "Surrogate predictions not computed"
        assert rep.design is not None, "Design points not set"

        # Verify shapes make sense
        assert rep.design.X.shape[0] == 4, f"Expected 4 design points, got {rep.design.X.shape[0]}"
        assert rep.grid.flat_grid.shape[0] == 25, f"Expected 5x5=25 grid points, got {rep.grid.flat_grid.shape[0]}"

        print("test_vsem_replicate_setup: PASSED")


def test_vsem_experiment_one_rep():
    """
    Test the full experiment pipeline with 1 replicate through the Experiment
    framework: setup, MCMC sampling (exact + approximate posteriors), rkpcn
    sampler, per-rep saving, and coverage computation.
    """
    key = jr.key(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / 'vsem_test'

        setup_kwargs = {
            'n_grid': 5,
            'n_design': 4,
            'surrogate_tag': 'gp',
        }
        run_kwargs = {
            'rkpcn_rho_vals': {'rkpcn90': 0.9},
            'mcmc_settings': {'n_samples': 10, 'n_burnin': 50, 'thin_window': 1},
            'rkpcn_settings': {'n_samples': 10, 'n_burnin': 50, 'thin_window': 1},
        }

        def subdir_name_fn(setup_kw, run_kw):
            return f'{setup_kw["surrogate_tag"]}_N{setup_kw["n_design"]}'

        experiment = Experiment(
            name='vsem_test',
            num_reps=1,
            base_out_dir=out_dir,
            base_key=key,
            Replicate=VSEMReplicate,
            subdir_name_fn=subdir_name_fn,
        )

        results, failed_reps, skipped_reps = experiment(
            run_kwargs=run_kwargs,
            setup_kwargs=setup_kwargs,
        )

        assert len(failed_reps) == 0, f"Replicates failed: {failed_reps}"

        # __call__ returns None now (per-rep saving)
        assert results[0] is None, "Expected None return from replicate"

        # Verify per-rep files were saved
        rep_dir = out_dir / 'gp_N4' / 'rep0'
        assert rep_dir.exists(), f"Rep directory not found: {rep_dir}"

        samples_path = rep_dir / 'samples.npz'
        assert samples_path.exists(), "samples.npz not saved"
        samples = dict(jnp.load(samples_path))
        expected_keys = {'exact', 'mean', 'eup', 'ep', 'rkpcn90'}
        assert expected_keys.issubset(samples.keys()), \
            f"Missing sample keys. Expected {expected_keys}, got {set(samples.keys())}"

        # Check sample shapes (n_samples=10, dim=2)
        for name in expected_keys:
            assert samples[name].shape == (10, 2), \
                f"Wrong shape for {name}: {samples[name].shape}"

        diagnostics_path = rep_dir / 'diagnostics.npz'
        assert diagnostics_path.exists(), "diagnostics.npz not saved"
        diag = dict(jnp.load(diagnostics_path))
        assert 'exact_accept_rate' in diag, "Missing exact_accept_rate"
        assert 'rkpcn90_accept_rate' in diag, "Missing rkpcn90_accept_rate"

        grid_dens_path = rep_dir / 'grid_densities.npz'
        assert grid_dens_path.exists(), "grid_densities.npz not saved"

        coverage_path = rep_dir / 'coverage.npz'
        assert coverage_path.exists(), "coverage.npz not saved"

        setup_path = rep_dir / 'setup_info.npz'
        assert setup_path.exists(), "setup_info.npz not saved"

        print("test_vsem_experiment_one_rep: PASSED")


if __name__ == '__main__':
    test_vsem_replicate_setup()
    test_vsem_experiment_one_rep()
    print("\nAll VSEM tests passed!")
