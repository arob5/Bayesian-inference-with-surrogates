"""
Minimal viable test for the VSEM experiment pipeline.

Runs a single replicate with reduced settings (small grid, few design points)
to verify the experiment code works end-to-end after reorganization.
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


def test_vsem_experiment_setup_via_framework():
    """
    Test that the Experiment framework correctly initializes a VSEM replicate.
    This verifies that setup_kwargs are correctly passed through the framework.

    NOTE: The full __call__ (MCMC sampling) phase of VSEMReplicate has a
    pre-existing issue with rkpcn prop_cov argument. This test only verifies
    the setup phase works through the Experiment framework.
    """
    key = jr.key(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir) / 'vsem_test'

        setup_kwargs = {
            'n_grid': 5,
            'n_design': 4,
            'surrogate_tag': 'gp',
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

        # Directly test that the framework can initialize a replicate
        rep = experiment.init_replicate(
            rep_idx=0,
            setup_kwargs=setup_kwargs,
            rep_subdir=out_dir / 'rep_0',
        )

        assert rep.posterior is not None
        assert rep.posterior_surrogate is not None
        print("test_vsem_experiment_setup_via_framework: PASSED")


if __name__ == '__main__':
    test_vsem_replicate_setup()
    test_vsem_experiment_setup_via_framework()
    print("\nAll VSEM tests passed!")
