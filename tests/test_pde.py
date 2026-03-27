"""
Minimal viable test for the elliptic PDE experiment pipeline.

Runs a single replicate with reduced settings (few design points, short MCMC
chains) to verify the experiment code works end-to-end.
This does NOT run the full paper experiment — see experiments/elliptic_pde/runner.py for that.
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
    "pde_experiment", REPO_ROOT / "experiments" / "elliptic_pde" / "experiment.py")
_pde_exp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pde_exp)
PDEReplicate = _pde_exp.PDEReplicate


def test_pde_replicate_setup():
    """
    Test that a single PDE replicate can be set up: generates the inverse
    problem, fits the surrogate GP, and builds the surrogate-based posterior.
    """
    key = jr.key(42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        rep = PDEReplicate(
            key=key,
            out_dir=out_dir,
            n_design=4,
            num_rff=50,
            write_to_file=True,
        )

        # Verify core components were created
        assert rep.posterior is not None, "Posterior not created"
        assert rep.posterior_surrogate is not None, "Surrogate posterior not created"
        assert rep.design is not None, "Design points not set"
        assert rep.ground_truth is not None, "Ground truth not stored"

        # Verify design shape
        assert rep.design.X.shape[0] == 4, f"Expected 4 design points, got {rep.design.X.shape[0]}"

        # Verify output files were written
        assert (out_dir / 'init_settings.npz').exists(), "init_settings.npz not written"
        assert (out_dir / 'design.npz').exists(), "design.npz not written"
        assert (out_dir / 'keys.npz').exists(), "keys.npz not written"

        print("test_pde_replicate_setup: PASSED")


def test_pde_replicate_run():
    """
    Test the full replicate pipeline: setup + MCMC sampling for exact and
    surrogate-based posteriors with minimal chain lengths.
    """
    key = jr.key(42)
    key_init, key_run = jr.split(key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = Path(tmp_dir)

        rep = PDEReplicate(
            key=key_init,
            out_dir=out_dir,
            n_design=4,
            num_rff=50,
            write_to_file=True,
        )

        rep(
            key=key_run,
            out_dir=out_dir,
            rho_vals=[0.9],
            mcmc_settings={'n_samples': 5, 'n_burnin': 10, 'thin_window': 1},
            mcwmh_settings={'n_chains': 2, 'n_samp_per_chain': 2,
                            'n_burnin': 10, 'thin_window': 1},
            rkpcn_settings={'n_samples': 5, 'n_burnin': 10, 'thin_window': 1},
        )

        # Verify samples file was written with expected keys
        assert (out_dir / 'samples.npz').exists(), "samples.npz not written"
        samples = dict(jnp.load(out_dir / 'samples.npz'))

        expected_keys = {'exact', 'mean', 'eup', 'ep_mcwmh', 'rkpcn90'}
        assert expected_keys <= set(samples.keys()), (
            f"Missing sample keys: {expected_keys - set(samples.keys())}"
        )

        # Verify diagnostics file was written with acceptance rates
        assert (out_dir / 'diagnostics.npz').exists(), "diagnostics.npz not written"
        diag = dict(jnp.load(out_dir / 'diagnostics.npz'))

        expected_diag_keys = {
            'exact_accept_rate', 'mean_accept_rate', 'eup_accept_rate',
            'rkpcn90_accept_rate',
        }
        assert expected_diag_keys <= set(diag.keys()), (
            f"Missing diagnostic keys: {expected_diag_keys - set(diag.keys())}"
        )

        # Acceptance rates should be between 0 and 1
        for k in expected_diag_keys:
            val = float(diag[k])
            assert 0.0 <= val <= 1.0, f"{k} = {val} not in [0, 1]"

        print("test_pde_replicate_run: PASSED")


if __name__ == '__main__':
    test_pde_replicate_setup()
    test_pde_replicate_run()
    print("\nAll PDE tests passed!")
