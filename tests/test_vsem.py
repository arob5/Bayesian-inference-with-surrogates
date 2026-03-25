"""
Minimal viable test for the VSEM experiment pipeline.

Runs a single replicate with reduced settings (small grid, few design points,
short MCMC chains) to verify the experiment code works end-to-end.
This does NOT run the full paper experiment — see experiments/vsem/runner.py for that.
"""
from jax import config
config.update('jax_enable_x64', True)

import sys
import tempfile
from pathlib import Path

import jax.random as jr

# Add experiments/vsem to path so we can import experiment.py
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'vsem'))

from experiment import VSEMReplicate, VSEMExperiment


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
    sampler, density comparison, and coverage computation.
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
            'rkpcn_rho_vals': {'rkpcn_0.9': 0.9},
            'mcmc_settings': {'n_samples': 10, 'n_burnin': 50, 'thin_window': 1},
            'rkpcn_settings': {'n_samples': 10, 'n_burnin': 50, 'thin_window': 1},
        }

        def subdir_name_fn(setup_kw, run_kw):
            return f'{setup_kw["surrogate_tag"]}_N{setup_kw["n_design"]}'

        experiment = VSEMExperiment(
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

        # Verify the replicate has expected outputs
        rep = results[0]
        assert rep.density_comparison is not None, "Density comparison not computed"
        assert rep.coverage_results is not None, "Coverage results not computed"
        assert rep.mcmc_results is not None, "MCMC results not stored"

        print("test_vsem_experiment_one_rep: PASSED")


if __name__ == '__main__':
    test_vsem_replicate_setup()
    test_vsem_experiment_one_rep()
    print("\nAll VSEM tests passed!")
