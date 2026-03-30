# Modular Uncertainty Propagation for Surrogate-Based Bayesian Inverse Problems

This repo contains the code for reproducing the experiments in the paper
[Propagating Surrogate Uncertainty in Bayesian Inverse Problems](https://www.arxiv.org/abs/2601.03532), as well as the review paper
[Surrogate-Based Bayesian Inference: Uncertainty Quantification and Active Learning](https://arxiv.org/abs/2603.13646). The primary dependencies are packages from the JAX ecosystem (JAX, NumPyro, GPjax, blackjax).

## Dependencies
To replicate the environment used to run the experiments:
```bash
uv python pin 3.13.8
uv sync --frozen
```

This will create a virtual environment; all Python code should be run using
this virtual environment, which can be activated using
```bash
source .venv/bin/activate
```

## Organization

### `uncprop/`
The core code underlying the experiments.
- `core/` — Inverse problem and probability distribution abstractions; MCMC sampling algorithms written in the [blackjax](https://blackjax-devs.github.io/blackjax/) style; `GPJaxSurrogate` wrapper around [gpjax](https://docs.jaxgaussianprocesses.com/); abstractions for random measures induced by GP surrogates.
- `models/` — Each model corresponds to a numerical experiment: an underlying mechanistic model, a statistical inverse problem, and a surrogate model. Contains `linear_Gaussian/`, `elliptic_pde/`, and `vsem/`.
- `utils/` — Experiment replicate framework; gpjax extensions (vectorized multi-output GP fitting); helpers for plotting, probability distributions, Wasserstein distance computation, and uniform grids.

### `experiments/`
Experiment runners, analysis scripts, and cluster submission scripts for each numerical experiment. Each subdirectory contains a `runner.py` entry point, an `experiment.py` defining the replicate logic, and shell scripts for cluster submission.

### `tests/`
Lightweight tests for each experiment that verify the full pipeline end-to-end with reduced settings.

## Running Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Reproducing Experiments

The experiments in the paper were run on the [Boston University Shared Computing Cluster (SCC)](https://www.bu.edu/tech/support/research/computing-resources/scc/), which uses the Sun Grid Engine (SGE) job scheduler. The submission scripts in `experiments/*/submit_runner.sh` are written for SGE. For clusters using other schedulers (e.g., Slurm), the SGE directives (`#$ ...`) would need to be replaced with the equivalent scheduler directives, but the Python commands are the same. All experiments can also be run locally (see below).

Each experiment consists of many independent replicates that are parallelized across cluster array jobs. Replicates that have already completed are automatically skipped on re-submission, and individual failed replicates can be re-run without repeating the entire experiment. PRNG keys are deterministically derived from a base seed and saved per-replicate, ensuring exact reproducibility.

### VSEM Experiment

The VSEM (Very Simple Ecosystem Model) experiment calibrates a 2-parameter ecosystem model using a GP surrogate of the log-posterior. It runs 100 replicates across 6 configurations: {`gp`, `clip_gp`} surrogate types × {4, 8, 16} design points.

#### Running on a cluster (SGE)

From the repo root:
```bash
cd experiments/vsem
qsub submit_runner.sh
```

This submits 60 array tasks (6 setups × 10 chunks of 10 replicates each). Each task runs a subset of replicates for one configuration. Output is saved per-replicate to `out/vsem/<setup_name>/rep<i>/`.

To monitor progress:
```bash
qstat -u $USER            # check job status
cat vsem_experiment.o*    # check logs
```

To check which replicates have completed:
```bash
cd experiments/vsem
python -c "
from experiment import check_completion_status
for setup in ['gp_N4', 'gp_N8', 'gp_N16', 'clip_gp_N4', 'clip_gp_N8', 'clip_gp_N16']:
    check_completion_status('../../out/vsem', setup, 100)
"
```

#### Re-running failed replicates

Individual replicates can be re-run using `main_manual()` in `runner.py`:
```bash
cd experiments/vsem
python -c "
from runner import main_manual
main_manual('vsem', 'gp', 4, 1e-4, [3, 17, 42], overwrite=True)
"
```

The arguments are: experiment name, surrogate tag (`'gp'` or `'clip_gp'`), number of design points, GP jitter, and a list of replicate indices to run. Setting `overwrite=True` re-runs even if output already exists.

#### Running locally

For local testing or running on a machine without a job scheduler:
```bash
cd experiments/vsem
python -c "
from runner import main_manual
main_manual('vsem', 'clip_gp', 4, 1e-4, [0, 1, 2], write_to_log_file=False)
"
```

A single replicate takes approximately 1–2 minutes on a modern laptop.

#### Post-hoc analysis

After all replicates have completed, generate all paper figures and diagnostics in a single step:
```bash
cd experiments/vsem
python analyze.py --experiment-name vsem
```

This produces:
- **Coverage/calibration plots** (`vsem_coverage_gp.pdf`, `vsem_coverage_clip_gp.pdf`): 3×3 grids showing coverage curves across design sizes and approximation methods
- **W2 distance box plots** (`w2_boxplots.pdf`, `w2_by_design_*.pdf`): Wasserstein-2 distance from each posterior approximation (exact, mean, EUP, RKPCN) to the grid-based expected posterior (EP)
- **Acceptance rate diagnostics**: printed to stdout for all samplers and configurations

All plots are saved to `out/vsem/`.

### Elliptic PDE Experiment

The elliptic PDE experiment solves a 1D diffusion inverse problem with a Karhunen–Loève parameterized log-permeability field (6 parameters). A batch independent GP surrogate of the forward model is fit, and posterior approximations are compared via MCMC. It runs 100 replicates across 3 design sizes: {10, 20, 30} design points.

#### Running on a cluster (SGE)

From the repo root:
```bash
cd experiments/elliptic_pde
qsub submit_runner.sh
```

This submits 30 array tasks (3 design sizes × 10 chunks of 10 replicates each). Each task runs a subset of replicates for one design size. Output is saved per-replicate to `out/pde_experiment/n_design_<N>/rep<i>/`.

To monitor progress:
```bash
qstat -u $USER
cat pde_experiment.o*
```

To check which replicates have completed:
```bash
cd experiments/elliptic_pde
python -c "
from analyze import check_completion_status
for n in [10, 20, 30]:
    completed, missing = check_completion_status('../../out/pde_experiment', n, 100)
    print(f'n_design={n}: {len(completed)} completed, {len(missing)} missing')
"
```

#### Re-running failed replicates

Individual replicates can be re-run using `main_manual()` in `runner.py`:
```bash
cd experiments/elliptic_pde
python -c "
from runner import main_manual
main_manual('pde_experiment', n_design=10, rep_idx=[3, 17], write_to_log_file=True, overwrite=True)
"
```

The arguments are: experiment name, number of design points, list of replicate indices to run, and whether to overwrite existing output.

#### Running locally

For local testing or running on a machine without a job scheduler:
```bash
cd experiments/elliptic_pde
python -c "
from runner import main_manual
main_manual('pde_local_test', n_design=4, rep_idx=[0], write_to_log_file=False)
"
```

A single replicate takes approximately 5–15 minutes depending on design size and hardware.

#### Post-hoc analysis

After all replicates have completed, generate all paper figures and diagnostics:
```bash
cd experiments/elliptic_pde
python analyze.py --experiment-name pde_experiment
```

This produces:
- **Coverage/calibration plots** (`pde_coverage.pdf`): Grid showing Mahalanobis ellipsoidal coverage curves across design sizes and approximation methods (mean, EUP, EP)
- **W2 distance box plots** (`pde_w2_ndesign_*.pdf`): Wasserstein-2 distance from each posterior approximation to the MCwMH expected posterior, one plot per design size
- **Acceptance rate diagnostics**: printed to stdout for all samplers and design sizes

All plots are saved to `out/pde_experiment/`.

#### Per-replicate diagnostics

To investigate the behavior of a specific replicate (e.g., trace plots, scatter plots):
```bash
cd experiments/elliptic_pde
python diagnose_rep.py --experiment-name pde_experiment --n-design 10 --rep 0 --output-dir ../../out/pde_experiment/diag
```

#### EP baseline validation

The expected posterior (EP) in the PDE experiment is approximated via Monte Carlo within Metropolis-Hastings (MCwMH) using random Fourier feature (RFF) approximated GP trajectories. The `validate_ep.py` script provides three studies to assess the quality of this approximation for specific replicates:

1. **MCwMH convergence**: Re-runs MCwMH at a heavier budget (500 chains × 200 samples) and compares to the standard budget (200 × 50) via W2 distance.
2. **RFF convergence**: Re-runs MCwMH with different numbers of random Fourier features (e.g., 500, 1000, 2000) and computes pairwise W2 distances to check that the RFF approximation has converged.
3. **Chain quality**: Post-hoc analysis of per-chain diagnostics (acceptance rates, ESS, final log-density). Identifies outlier chains, visualizes EP samples colored by chain, and reports W2 between full and filtered EP samples.

Run all three studies for a specific replicate:
```bash
cd experiments/elliptic_pde
qsub submit_validate.sh
```

Or run individual studies (chain quality is fast enough for interactive use):
```bash
qsub -v STUDIES=chain_quality submit_validate.sh
qsub -v STUDIES=mcwmh_convergence submit_validate.sh
qsub -v STUDIES=rff_convergence submit_validate.sh
```

Configure which replicate to validate by editing `submit_validate.sh` or passing arguments directly:
```bash
python validate_ep.py \
    --experiment-name pde_experiment \
    --n-design 10 --rep 0 \
    --output-dir ../../out/pde_experiment/validation \
    --studies chain_quality mcwmh_convergence rff_convergence
```

Output (plots and `.npz` files) is saved to the specified output directory.

### Linear Gaussian Experiment

The linear Gaussian experiment provides a closed-form test case where all posterior approximations (exact, mean plug-in, EUP, EP) are available analytically. The parameter space is 100-dimensional with a linear forward model (Gaussian deconvolution) and Gaussian prior. The experiment runs 100 replicates with a calibrated surrogate (Q = G C₀ Gᵀ).

This experiment requires the `modmcmc` package (for MCMC-based EP comparisons), which is not included in the default environment. The analytical comparisons (coverage, KL divergence, Wasserstein distance) do not require `modmcmc`.

#### Running the experiment

From the repo root:
```bash
cd experiments/linear_Gaussian
python runner.py
```

This runs 100 replicates of the coverage test with the default settings (d=100, noise_sd=0.2, every 4th index observed). Results are saved to `out/` within the experiment directory.

#### Running locally with custom settings

The experiment can also be run from a notebook or script:
```python
from uncprop.models.linear_Gaussian.inverse_problem_setup import make_inverse_problem
from uncprop.models.linear_Gaussian.Gaussian import Gaussian
from runner import run_coverage_test
import numpy as np

rng = np.random.default_rng(532124)
inv_prob, g_conv, grid, idx_obs = make_inverse_problem(
    rng=rng, d=100, noise_sd=0.2,
    ker_length=21, ker_lengthscale=20, s=4)
Q = inv_prob.G @ inv_prob.prior.cov @ inv_prob.G.T
tests, res, probs = run_coverage_test(
    rng, n_reps=10, m0=inv_prob.prior.mean,
    C0=inv_prob.prior.cov, Sig=inv_prob.noise.cov,
    G=inv_prob.G, Q_true=Q, Q=Q, include_mcmc=False)
```

Setting `include_mcmc=False` skips the MCMC-based EP approximations (which require `modmcmc`) and only computes the analytical comparisons. The notebooks in `experiments/linear_Gaussian/` provide interactive exploration of the results.
