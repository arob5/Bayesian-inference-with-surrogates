# RKPCN Benchmark Framework

A shared framework for comparing RKPCN algorithm variants across
replicates and experiments. Supports VSEM and elliptic-PDE replicates
through a common `SurrogateDistribution` interface; new experiments
can be added by implementing a `reconstruct_replicate` loader and
registering it in `replicate_loader.py`.

This directory is intended for *algorithm ablation* work — things
like tempering schedules, support-point strategies, and new proposal
variants — before any candidate is promoted to the paper experiment
scripts. The scripts here do not touch the main experiment outputs
(`out/vsem/`, `out/elliptic_pde/`); benchmark outputs land under
`out/benchmarks/`.

## Contents

- `benchmark.py` — main CLI entry point (`run`, `analyze`).
- `replicate_loader.py` — dispatches to experiment-specific replicate
  loaders; defines the common replicate API.
- `diagnostics.py` — summary tables, ACF / IAT, VSEM-only W2 vs. EP.
- `plots.py` — generic corner / trace / ACF / marginal plots.
- `submit_benchmark.sh`, `submit_analyze.sh`, `submit_smoke_test.sh`
  — SCC (SGE) job submission wrappers.
- `multichain_sweep.yaml` — canonical rho × u-steps sweep.
- `smoke_test.yaml`, `smoke_test_pde.yaml` — cheap end-to-end tests.

## 1. Writing a benchmark config

Each config is a YAML with three top-level keys.

```yaml
common_kwargs:          # applied to every variant (per-variant keys override)
  n_total: 55000
  n_burnin: 50000
  n_chains: 4
  adaptive: true
  adapt_interval: 50
  target_accept: 0.234
  weight_method: pritchard       # equal | mean_logdens | pritchard
  init_method: ep_direct_sampling  # ep_direct_sampling | uniform_support

replicates:             # run EVERY variant on EACH of these replicates
  - {experiment: vsem,         setup: clip_gp_N4,  rep: 0}
  - {experiment: vsem,         setup: clip_gp_N4,  rep: 1}
  - {experiment: vsem,         setup: clip_gp_N8,  rep: 0}
  - {experiment: elliptic_pde, setup: n_design_10, rep: 0}

variants:               # the algorithm configurations to compare
  - {label: rho90_4ch_u1,  rho: 0.9}
  - {label: rho99_4ch_u20, rho: 0.99, n_u_steps: 20}
  # ...etc
```

**`replicates` entries** must specify `experiment`, `setup`, `rep`.
- `experiment`: `vsem` or `elliptic_pde`.
- `setup`: the subdirectory under `out/<experiment>/`. VSEM uses
  `clip_gp_N<N>` / `gp_N<N>`; PDE uses `n_design_<N>`.
- `rep`: integer replicate index.

The referenced experiment output must already exist on disk (i.e.
`out/vsem/clip_gp_N4/rep0/` must contain the saved replicate
artifacts). The framework *reconstructs* the replicate on the fly
from the base PRNG key and saved init settings — it does not re-run
the original experiment.

**`variants` entries** must carry a unique `label`. All other fields
are kwargs forwarded to `run_rkpcn_chain` (single-chain) or
`run_rkpcn_multi_chain` (when `n_chains > 1`). Common overrides:
`rho`, `n_u_steps`, `adaptive`, `prop_cov_scale`, `init_method`,
`weight_method`, `n_chains`.

## 2. Running a benchmark via qsub

From `experiments/benchmarks/`:

```bash
qsub submit_benchmark.sh                                     # default config
qsub -v CONFIG=my_ablation.yaml submit_benchmark.sh
qsub -v CONFIG=my_ablation.yaml,SAVE_TRACE=1 submit_benchmark.sh
qsub -v OUTPUT_DIR=/abs/path/to/run submit_benchmark.sh     # skip auto-naming
```

Environment variables the submit script reads:

- `CONFIG` — YAML config path (relative to `experiments/benchmarks/`).
  Default `multichain_sweep.yaml`.
- `OUTPUT_DIR` — absolute output directory. Unset → auto-named
  `out/benchmarks/<config_basename>_<UTC_timestamp>/`.
- `OUTPUT_SUFFIX` — appended to the auto-name if `OUTPUT_DIR` is
  unset. Useful for labelling ablations.
- `SEED` — PRNG seed (default 42).
- `SAVE_TRACE` — `"1"` to save full unthinned single-chain traces
  (default off; multi-chain never saves full trace).

**Smoke test first.** The full sweep can run for hours. Before
committing, do:

```bash
qsub submit_smoke_test.sh                                     # VSEM
qsub -v CONFIG=smoke_test_pde.yaml submit_smoke_test.sh       # PDE
```

These run a single small 4-chain variant for ~2-5 min and validate
that replicate reconstruction, kernel construction, EP-direct init,
mode detection, and output saving all work end-to-end.

## 3. Where outputs land

Auto-generated (unless `OUTPUT_DIR` is overridden):

```
out/benchmarks/<config_basename>_<UTC_timestamp>[_<SUFFIX>]/
  benchmark_manifest.json      # config + git SHA + timestamps + status
  <experiment>_<setup>_rep<N>/
    benchmark_manifest.json    # not present (top-level only)
    <variant_label>/
      samples.npz              # thinned post-burnin + (multi-chain) per-chain
      summary.json             # ESS, accept rate, IAT, R-hat, mode weights, ...
      trace.npz                # optional, single-chain with --save-trace
    ...
```

`samples.npz` fields depend on the variant:
- Always: `post_burnin` (thinned pooled samples).
- Multi-chain: `sample_weights`, `chain<m>_post_burnin`, `mode_weights`,
  `mode_labels`, `init_positions`.

`summary.json` mirrors these plus per-chain convergence diagnostics
(R-hat, discarded burn-in, failure reasons) for multi-chain runs.

The top-level `benchmark_manifest.json` records the config, git SHA,
seed, and timestamps. The analyze step later writes an
`analysis_manifest.json` next to it.

## 4. Analyzing results

From `experiments/benchmarks/`:

```bash
qsub -v RUN_DIR=/abs/path/to/run submit_analyze.sh
```

Or interactively:

```bash
python -m benchmark analyze --output-dir /abs/path/to/run
```

The analyze step prints a banner that includes the absolute run
directory, manifest fields (timestamp, git SHA, status), and per-rep
counts — so there's no ambiguity about which benchmark is being
analyzed.

For each replicate under the run directory, it produces:

- Printed summary table (ESS, accept rate, IAT, per-dim means).
- Per-variant **corner plot** (`corner_<label>.pdf`) — univariate
  marginals on the diagonal, pairwise 2D scatters on the
  off-diagonals, mode-colored for multi-chain runs, with init-position
  stars.
- **Cross-variant marginals overview** (`marginals_overview.pdf`) —
  all variants' univariate marginals overlaid per dimension.
- **Trace plots** per variant (`trace_<label>.pdf`) — multi-chain
  traces colored by mode (when per-chain positions are saved).
- **ACF** of log-density per variant (`acf_<label>.pdf`).

VSEM-specific extras (only when a 2D grid-based EP density is
available for the replicate):

- W2 distance from each variant's KDE to the grid-based EP (printed).
- 2D scatter plots overlaid on EP contours
  (`scatter_vs_ep.pdf`, `scatter_vs_ep_annotated.pdf`).
- Density heatmaps for exact / mean / EUP / EP
  (`density_heatmaps.pdf`).

**PDE replicates** currently get only the generic plots — there is no
EP baseline for the PDE problem yet. Once an EP baseline is
implemented, the corresponding W2 and contour plots will drop into
place automatically without further changes to this framework.

## 5. Interactive use

You can import any piece directly:

```python
import sys
sys.path.insert(0, 'experiments/benchmarks')

from replicate_loader import load_replicate
from uncprop.core.rkpcn_multichain import run_rkpcn_multi_chain

rep, _ = load_replicate('vsem', 'clip_gp_N4', rep=0)
result = run_rkpcn_multi_chain(
    key=jr.key(0),
    surrogate_post=rep.posterior_surrogate,
    n_chains=4, rho=0.9, n_u_steps=1,
    n_total=10_000, n_burnin=8_000, adaptive=True,
)
```

## 6. Adding a new experiment

1. Create `experiments/<my_experiment>/reconstruct.py` with a
   `reconstruct_replicate(base_dir, setup_name, rep_idx, ...)` that
   returns `(rep_obj, key_run)`. Mirror
   `experiments/vsem/reconstruct.py` or
   `experiments/elliptic_pde/reconstruct.py`.
2. The `rep_obj` must expose `.posterior` (a `Posterior`) and
   `.posterior_surrogate` (a `SurrogateDistribution`).
3. Add a dispatch branch in `replicate_loader.py:load_replicate`.
4. Add experiment-specific plots to
   `experiments/<my_experiment>/plots.py` and gate them in
   `benchmark.py:analyze_benchmark` the same way the VSEM
   2D-scatter plots are gated behind `has_grid(rep)`.
