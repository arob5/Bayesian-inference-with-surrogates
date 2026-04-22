# experiments/benchmarks/benchmark.py
"""
Multi-replicate, multi-variant RKPCN benchmark.

This module is the single entry point for algorithm-comparison
experiments. It sweeps over a list of replicates (possibly spanning
experiments — VSEM, PDE, ...) and, for each replicate, runs a list
of RKPCN variants. Results are written to a canonical on-disk layout
that the ``analyze`` subcommand reads back.

Usage
-----
Run a benchmark::

    cd experiments/benchmarks
    python -m benchmark run --config my_sweep.yaml --output-dir <DIR>

Analyze an existing run::

    python -m benchmark analyze --output-dir <DIR>

YAML schema
-----------
::

    common_kwargs:        # applied to every variant (overridden per-variant)
      n_total: 55000
      n_burnin: 50000
      n_chains: 4
      adaptive: true
      ...

    replicates:           # required; run each variant on each rep
      - {experiment: vsem, setup: clip_gp_N4, rep: 0}
      - {experiment: vsem, setup: clip_gp_N8, rep: 2}
      - {experiment: elliptic_pde, setup: n_design_10, rep: 0}

    variants:
      - {label: rho90_4ch, rho: 0.9}
      - {label: rho99_4ch, rho: 0.99}

Backward-compat CLI flags ``--experiment/--setup/--rep`` are accepted;
they build a single-replicate list at run time (overriding the YAML).

Output directory layout
-----------------------
::

    <output_dir>/
      benchmark_manifest.json            # config + timestamp + git SHA
      <rep_id>/                          # e.g. "vsem_clip_gp_N4_rep0"
        <variant_label>/
          samples.npz
          summary.json
          trace.npz                      # only if --save-trace

``<rep_id>`` is built by
:func:`experiments.benchmarks.replicate_loader.rep_id`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'experiments'))

# Package-style imports are awkward because experiments/ is not a
# Python package. Use path-based imports:
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replicate_loader import (
    load_replicate, rep_id, has_grid, get_par_names, load_vsem_saved_data,
)
from diagnostics import (
    summary_table, w2_vs_grid_ep, integrated_autocorrelation_time,
)
import plots as bench_plots

from uncprop.core.rkpcn_multichain import (
    run_rkpcn_chain, run_rkpcn_multi_chain,
)
from uncprop.core.samplers import get_adapted_proposal


# =============================================================================
# Run: single variant, single replicate
# =============================================================================

def _dispatch_run_variant(
    key,
    rep,
    variant_kwargs: dict,
    default_prop_cov,
) -> dict:
    """Dispatch a variant run based on its kwargs.

    ``variant_kwargs`` is consumed destructively; callers should copy.
    """
    label = variant_kwargs.pop('label', 'variant')
    n_chains = variant_kwargs.pop('n_chains', 1)
    adaptive = variant_kwargs.pop('adaptive', False)
    v_prop_cov = variant_kwargs.pop('prop_cov', default_prop_cov)

    # prop_cov_scale shorthand
    scale = variant_kwargs.pop('prop_cov_scale', None)
    if scale is not None:
        v_prop_cov = float(scale) ** 2 * default_prop_cov

    surrogate_post = rep.posterior_surrogate
    par_names = get_par_names(rep)

    if n_chains > 1:
        return run_rkpcn_multi_chain(
            key=key,
            surrogate_post=surrogate_post,
            n_chains=n_chains,
            prop_cov=v_prop_cov,
            label=label,
            adaptive=adaptive,
            par_names=par_names,
            **variant_kwargs,
        )
    return run_rkpcn_chain(
        key=key,
        surrogate_post=surrogate_post,
        prop_cov=v_prop_cov,
        label=label,
        adaptive=adaptive,
        **variant_kwargs,
    )


def _save_variant_output(var_dir: Path, result: dict, save_trace: bool):
    """Persist a variant result: samples.npz + summary.json (+ trace.npz)."""
    var_dir.mkdir(parents=True, exist_ok=True)

    # ---- samples.npz ----
    post = result['post_burnin']
    thin = max(1, post.shape[0] // 2000)
    save_dict = {'post_burnin': post[::thin]}

    if result.get('sample_weights') is not None:
        sw = np.asarray(result['sample_weights'])[::thin]
        tot = sw.sum()
        if tot > 0:
            sw = sw / tot
        save_dict['sample_weights'] = sw

    per_chain = result.get('per_chain_results')
    if per_chain is not None:
        for m, cr in enumerate(per_chain):
            save_dict[f'chain{m}_post_burnin'] = cr['post_burnin'][::thin]
        if result.get('mode_weights') is not None:
            save_dict['mode_weights'] = result['mode_weights']
        if result.get('mode_labels') is not None:
            save_dict['mode_labels'] = result['mode_labels']
        if result.get('init_positions') is not None:
            save_dict['init_positions'] = result['init_positions']

    np.savez(var_dir / 'samples.npz', **save_dict)

    # ---- trace.npz (optional) ----
    if save_trace and 'positions' in result and not result.get('multi_chain', False):
        np.savez(var_dir / 'trace.npz',
                 positions=result['positions'],
                 logdensities=result['logdensities'],
                 accept_probs=result['accept_probs'])

    # ---- summary.json ----
    ld = np.asarray(result.get('logdensities', []))
    n_burnin = int(result.get('n_burnin', 0))
    ld_post = ld[n_burnin:] if ld.size > 0 else np.empty(0)
    iat_ld = (integrated_autocorrelation_time(ld_post)
              if ld_post.size > 1 else float('nan'))

    summary = {
        'label': result.get('label'),
        'rho': float(result.get('rho', 0)),
        'n_u_steps': int(result.get('n_u_steps', 1)),
        'adaptive': bool(result.get('adaptive', False)),
        'multi_chain': bool(result.get('multi_chain', False)),
        'n_chains': int(result.get('n_chains', 1)),
        'accept_rate': float(result.get('accept_rate', 0)),
        'min_ess': float(min(result['ess'])),
        'ess': [float(e) for e in result['ess']],
        'iat_logdensity': float(iat_ld),
        'n_post_burnin': int(post.shape[0]),
        'n_burnin': n_burnin,
        'runtime': float(result.get('runtime', 0)),
    }
    if result.get('mode_weights') is not None:
        summary['mode_weights'] = [float(w) for w in result['mode_weights']]
        summary['mode_membership'] = [list(map(int, ms))
                                       for ms in result['mode_membership']]
        summary['n_failed_chains'] = int(result['failed_mask'].sum())
        summary['n_modes'] = int(len(result['mode_weights']))
        summary['weight_method'] = result.get('weight_method', 'unknown')
        summary['init_positions'] = np.asarray(result['init_positions']).tolist()
        summary['per_chain_convergence'] = [
            {
                'chain_idx': i,
                'converged': bool(cr.get('convergence', {})
                                  .get('converged', False)),
                'rhat': float(cr.get('convergence', {}).get('rhat', float('nan'))),
                'n_discarded': int(cr.get('convergence', {})
                                   .get('n_discarded', 0)),
                'n_kept': int(cr.get('convergence', {}).get('n_kept', 0)),
                'fail_reason': cr.get('convergence', {}).get('fail_reason'),
                'mode_label': int(result['mode_labels'][i]),
                'accept_rate': float(cr['accept_rate']),
                'min_ess': float(min(cr['ess'])),
            }
            for i, cr in enumerate(result['per_chain_results'])
        ]
    if result.get('adapted_prop_cov_diag'):
        summary['adapted_prop_cov_diag'] = result['adapted_prop_cov_diag']
    with open(var_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Run subcommand
# =============================================================================

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return 'unknown'


def run_benchmark(
    config: dict,
    output_dir: Path,
    seed: int = 42,
    save_trace: bool = False,
):
    """Run all (replicate, variant) combinations and persist to disk.

    Parameters
    ----------
    config : dict with keys 'replicates', 'variants', 'common_kwargs' (opt).
    output_dir : Path
        Root output directory; per-replicate subdirs created under it.
    seed : int
        Base PRNG seed for variant keys.
    save_trace : bool
        If True, save full (unthinned) single-chain traces per variant.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replicates = config['replicates']
    variants = config['variants']
    common_kwargs = config.get('common_kwargs', {})

    # ---- Manifest (written up-front so partial runs are debuggable) ----
    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_sha': _git_sha(),
        'seed': seed,
        'save_trace': save_trace,
        'n_replicates': len(replicates),
        'n_variants': len(variants),
        'replicates': replicates,
        'variants': variants,
        'common_kwargs': {k: str(v) for k, v in common_kwargs.items()},
        'status': 'in_progress',
    }
    manifest_path = output_dir / 'benchmark_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'Manifest: {manifest_path}')

    # ---- Outer loop: replicates ----
    top_key = jr.key(seed)
    for rep_i, rep_spec in enumerate(replicates):
        exp = rep_spec['experiment']
        setup = rep_spec['setup']
        rep_idx = int(rep_spec['rep'])

        rid = rep_id(exp, setup, rep_idx)
        print('\n' + '=' * 72)
        print(f'Replicate {rep_i + 1}/{len(replicates)}: {rid}')
        print('=' * 72)

        try:
            rep, _ = load_replicate(exp, setup, rep_idx)
        except Exception as e:
            print(f'  FAILED to load replicate: {e}')
            continue

        rep_out = output_dir / rid
        rep_out.mkdir(parents=True, exist_ok=True)

        # Compute a default proposal covariance on the exact posterior.
        # We reuse this across all variants for this replicate.
        print('  Computing adapted proposal covariance '
              '(via exact-posterior MCMC)...')
        top_key, key_adapt = jr.split(top_key)
        default_prop_cov, _ = get_adapted_proposal(key_adapt, rep.posterior)
        print(f'    diag: {np.diag(np.array(default_prop_cov))}')

        # ---- Inner loop: variants ----
        for v_i, variant in enumerate(variants):
            jax.clear_caches()
            top_key, subkey = jr.split(top_key)

            kwargs = dict(common_kwargs)
            kwargs.update(variant)
            label = kwargs.get('label', f'variant_{v_i}')
            print(f'\n[rep {rep_i+1}/{len(replicates)}, '
                  f'variant {v_i+1}/{len(variants)}] {label}')

            try:
                result = _dispatch_run_variant(
                    subkey, rep, kwargs, default_prop_cov)
            except Exception as e:
                print(f'  FAILED: {e}')
                continue

            _save_variant_output(rep_out / label, result, save_trace)

    # ---- Finalize manifest ----
    manifest['status'] = 'complete'
    manifest['completed_utc'] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nBenchmark complete. Output: {output_dir}')


# =============================================================================
# Analyze subcommand
# =============================================================================

def _load_rep_results(rep_dir: Path) -> dict:
    """Load all variant results under one replicate subdir."""
    results = {}
    for var_dir in sorted(rep_dir.iterdir()):
        if not var_dir.is_dir():
            continue
        summary_path = var_dir / 'summary.json'
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        label = summary['label']

        entry = {'summary': summary, 'label': label}

        # Samples
        samples_path = var_dir / 'samples.npz'
        if samples_path.exists():
            data = np.load(samples_path)
            entry['post_burnin'] = data['post_burnin']
            if 'sample_weights' in data.files:
                entry['sample_weights'] = data['sample_weights']

            chain_keys = sorted(
                [k for k in data.files
                 if k.startswith('chain') and k.endswith('_post_burnin')],
                key=lambda s: int(s.replace('chain', '')
                                  .replace('_post_burnin', '')))
            if chain_keys:
                entry['per_chain_results'] = [
                    {'post_burnin': data[ck]} for ck in chain_keys]
                if 'mode_weights' in data.files:
                    entry['mode_weights'] = data['mode_weights']
                if 'mode_labels' in data.files:
                    entry['mode_labels'] = data['mode_labels']
                if 'init_positions' in data.files:
                    entry['init_positions'] = data['init_positions']

        # Trace (optional)
        trace_path = var_dir / 'trace.npz'
        if trace_path.exists():
            t = np.load(trace_path)
            entry['positions'] = t['positions']
            entry['logdensities'] = t['logdensities']
            entry['accept_probs'] = t['accept_probs']
            entry['n_burnin'] = summary['n_burnin']
        else:
            entry['logdensities'] = np.empty(0)
            entry['n_burnin'] = 0

        # Fold scalar fields the plot functions expect
        entry['rho'] = summary.get('rho')
        entry['accept_rate'] = summary.get('accept_rate')
        entry['ess'] = np.asarray(summary.get('ess', []))

        results[label] = entry
    return results


def analyze_benchmark(output_dir: Path, save_plots: bool = True,
                     thin: int = 5):
    """Analyze all replicates under a benchmark run directory.

    Prints a startup banner with absolute paths, loads the manifest,
    and for each replicate: loads its results, prints a summary table,
    and produces per-variant corner plots, trace plots, an ACF plot,
    plus VSEM-specific 2D scatter/contour plots when applicable.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir).resolve()
    print('=' * 72)
    print('Benchmark Analysis')
    print(f'  run directory: {output_dir}')

    manifest_path = output_dir / 'benchmark_manifest.json'
    if not manifest_path.exists():
        print(f'  WARN: no benchmark_manifest.json at {manifest_path}')
        manifest = None
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f'  timestamp_utc: {manifest.get("timestamp_utc")}')
        print(f'  git_sha:       {manifest.get("git_sha")}')
        print(f'  status:        {manifest.get("status")}')
        print(f'  n_replicates:  {manifest.get("n_replicates")}')
        print(f'  n_variants:    {manifest.get("n_variants")}')
    print('=' * 72)

    # Iterate over replicate subdirs (discover from filesystem to
    # survive partial runs).
    rep_dirs = sorted(
        [p for p in output_dir.iterdir()
         if p.is_dir() and (p / 'benchmark_manifest.json').exists() is False])
    if not rep_dirs:
        # Fall back to all subdirs
        rep_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir()])

    analysis_manifest = {
        'run_dir': str(output_dir),
        'analyzed_utc': datetime.now(timezone.utc).isoformat(),
        'replicate_dirs': [],
    }

    for rep_dir in rep_dirs:
        if rep_dir.name == '__pycache__':
            continue
        # Parse experiment / setup / rep from rep_dir name.
        # Format: "<experiment>_<setup>_rep<idx>".
        name = rep_dir.name
        if '_rep' not in name:
            continue
        try:
            prefix, rep_str = name.rsplit('_rep', 1)
            rep_idx = int(rep_str)
        except ValueError:
            continue
        # Experiment is the first token; setup is the remainder.
        if '_' in prefix:
            experiment, setup = prefix.split('_', 1)
        else:
            experiment, setup = prefix, ''

        print('\n' + '-' * 72)
        print(f'Analyzing replicate: {name}')
        print(f'  experiment={experiment}, setup={setup}, rep={rep_idx}')
        print('-' * 72)

        # Load results for this rep
        results = _load_rep_results(rep_dir)
        if not results:
            print('  no variant results found')
            continue
        print(f'  {len(results)} variant(s) found')

        # Try loading the replicate (for par_names, grid, reference samples).
        par_names = None
        grid = None
        ep_density = None
        reference_samples = None
        grid_densities = None
        vsem_data = None

        try:
            rep_obj, _ = load_replicate(experiment, setup, rep_idx)
            par_names = get_par_names(rep_obj)
            if has_grid(rep_obj):
                grid = rep_obj.grid
                # Try to load the saved grid densities + reference samples
                try:
                    base = REPO_ROOT / 'out' / experiment
                    vsem_data = load_vsem_saved_data(base, setup, rep_idx)
                    if vsem_data['grid_densities'] is not None:
                        grid_densities = vsem_data['grid_densities']
                        ep_density = grid_densities.get('ep')
                    if vsem_data['samples'] is not None:
                        reference_samples = {}
                        for nm in ('exact', 'mean', 'eup'):
                            if nm in vsem_data['samples']:
                                reference_samples[nm] = np.array(
                                    vsem_data['samples'][nm])
                except Exception as e:
                    print(f'  VSEM side data not loaded: {e}')
        except Exception as e:
            print(f'  replicate reload failed (analysis will proceed '
                  f'with reduced features): {e}')

        # ---- Summary table ----
        print('\n  ~~ Summary ~~')
        summary_table(results, par_names=par_names)

        # ---- W2 vs. EP (VSEM only) ----
        w2_results = {}
        if ep_density is not None and grid is not None:
            print('\n  ~~ W2 to EP (grid-based) ~~')
            w2_results = w2_vs_grid_ep(results, ep_density, grid, thin=thin)

        # ---- Per-variant corner plots ----
        print('\n  ~~ Corner plots ~~')
        for label, res in results.items():
            fig, _ = bench_plots.plot_corner(
                post_burnin=res['post_burnin'],
                sample_weights=res.get('sample_weights'),
                per_chain_post_burnin=[c['post_burnin']
                                        for c in (res.get('per_chain_results')
                                                  or [])] or None,
                mode_labels=res.get('mode_labels'),
                init_positions=res.get('init_positions'),
                par_names=par_names,
                title=f'{name} — {label}',
                thin=thin,
            )
            if save_plots:
                out = rep_dir / f'corner_{label}.pdf'
                fig.savefig(out, bbox_inches='tight', dpi=150)
                print(f'    saved: {out.name}')
            plt.close(fig)

        # ---- Cross-variant univariate marginals overview ----
        fig, _ = bench_plots.plot_marginals_overview(
            results, par_names=par_names)
        if save_plots:
            out = rep_dir / 'marginals_overview.pdf'
            fig.savefig(out, bbox_inches='tight', dpi=150)
            print(f'    saved: {out.name}')
        plt.close(fig)

        # ---- Per-variant traces (if traces saved) ----
        has_any_trace = any(r.get('positions') is not None
                            or r.get('per_chain_results')
                            for r in results.values())
        if has_any_trace:
            for label, res in results.items():
                if res.get('positions') is None and not res.get('per_chain_results'):
                    continue
                fig, _ = bench_plots.plot_traces(res, par_names=par_names)
                if save_plots:
                    out = rep_dir / f'trace_{label}.pdf'
                    fig.savefig(out, bbox_inches='tight', dpi=150)
                plt.close(fig)

        # ---- ACF (per-variant) ----
        for label, res in results.items():
            if res.get('logdensities') is None or res['logdensities'].size < 2:
                continue
            fig, _ = bench_plots.plot_acf(res)
            if save_plots:
                out = rep_dir / f'acf_{label}.pdf'
                fig.savefig(out, bbox_inches='tight', dpi=150)
            plt.close(fig)

        # ---- VSEM-specific 2D plots ----
        if grid is not None and ep_density is not None:
            sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'vsem'))
            import plots as vsem_plots

            print('\n  ~~ VSEM 2D scatter plots ~~')
            fig, _ = vsem_plots.plot_samples_vs_ep(
                results, ep_density, grid, thin=thin,
                reference_samples=reference_samples)
            if save_plots:
                out = rep_dir / 'scatter_vs_ep.pdf'
                fig.savefig(out, bbox_inches='tight', dpi=150)
                print(f'    saved: {out.name}')
            plt.close(fig)

            fig, _ = vsem_plots.plot_samples_vs_ep_annotated(
                results, ep_density, grid, thin=thin,
                reference_samples=reference_samples)
            if save_plots:
                out = rep_dir / 'scatter_vs_ep_annotated.pdf'
                fig.savefig(out, bbox_inches='tight', dpi=150)
                print(f'    saved: {out.name}')
            plt.close(fig)

            if grid_densities is not None:
                available = [n for n in ('exact', 'mean', 'eup', 'ep')
                             if n in grid_densities]
                if available:
                    fig, _ = vsem_plots.plot_density_heatmaps(
                        grid_densities, grid, names=available)
                    if save_plots:
                        out = rep_dir / 'density_heatmaps.pdf'
                        fig.savefig(out, bbox_inches='tight', dpi=150)
                        print(f'    saved: {out.name}')
                    plt.close(fig)

        # Record for analysis manifest
        analysis_manifest['replicate_dirs'].append({
            'rep_dir': str(rep_dir),
            'experiment': experiment,
            'setup': setup,
            'rep': rep_idx,
            'n_variants': len(results),
            'has_grid_analysis': (grid is not None
                                  and ep_density is not None),
            'w2_available': len(w2_results) > 0,
        })

    # Write analysis manifest
    with open(output_dir / 'analysis_manifest.json', 'w') as f:
        json.dump(analysis_manifest, f, indent=2)
    print(f'\nAnalysis manifest: {output_dir / "analysis_manifest.json"}')


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-replicate RKPCN benchmark: run variants or '
                    'analyze results.')
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- run ----
    p_run = sub.add_parser('run', help='Run benchmark variants.')
    p_run.add_argument('--config', type=str, required=True,
                       help='YAML config (see module docstring).')
    p_run.add_argument('--output-dir', type=str, required=True)
    p_run.add_argument('--seed', type=int, default=42)
    p_run.add_argument('--save-trace', action='store_true')
    # Convenience: single-rep override.
    p_run.add_argument('--experiment', type=str, default=None,
                       help='Overrides YAML replicates with a single rep.')
    p_run.add_argument('--setup', type=str, default=None)
    p_run.add_argument('--rep', type=int, default=None)

    # ---- analyze ----
    p_an = sub.add_parser('analyze', help='Analyze an existing benchmark run.')
    p_an.add_argument('--output-dir', type=str, required=True,
                      help='Benchmark run directory (containing '
                           'benchmark_manifest.json).')
    p_an.add_argument('--thin', type=int, default=5)
    p_an.add_argument('--no-save', action='store_true',
                      help='Skip saving plots to disk.')

    args = parser.parse_args()

    if args.command == 'run':
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)

        # CLI override for single-rep ergonomics
        if args.experiment is not None or args.setup is not None or args.rep is not None:
            missing = [nm for nm, v in
                       [('--experiment', args.experiment),
                        ('--setup', args.setup),
                        ('--rep', args.rep)]
                       if v is None]
            if missing:
                parser.error(f'Single-rep override requires all of: '
                             f'{missing}')
            config['replicates'] = [{
                'experiment': args.experiment,
                'setup': args.setup,
                'rep': args.rep,
            }]

        if 'replicates' not in config or not config['replicates']:
            parser.error('Config must contain a non-empty "replicates" list '
                         '(or use --experiment/--setup/--rep).')
        if 'variants' not in config or not config['variants']:
            parser.error('Config must contain a non-empty "variants" list.')

        run_benchmark(config=config, output_dir=Path(args.output_dir),
                      seed=args.seed, save_trace=args.save_trace)

    elif args.command == 'analyze':
        analyze_benchmark(Path(args.output_dir), save_plots=not args.no_save,
                          thin=args.thin)


if __name__ == '__main__':
    main()
