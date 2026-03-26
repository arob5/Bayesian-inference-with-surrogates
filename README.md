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

*(Directions to be added.)*

### Linear Gaussian Experiment

*(Directions to be added.)*
