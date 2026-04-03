# experiments/vsem/rkpcn_analysis/benchmark.py
"""
Benchmark framework for comparing RKPCN algorithm variants.

Workflow:
1. Define variants as a list of dicts specifying algorithm settings.
2. Run ``run_benchmark()`` to execute all variants and save results to disk.
   This can be run as a batch job on the cluster.
3. Load results with ``load_benchmark_results()`` and analyze with
   ``print_benchmark_summary()`` and ``plot_benchmark_comparison()``.

Usage (CLI)::

    # Run benchmark (e.g., from experiments/vsem/)
    python -m rkpcn_analysis.benchmark \\
        --experiment-name vsem \\
        --setup clip_gp_N4 --rep 0 \\
        --config benchmark_variants.yaml \\
        --output-dir ../../out/vsem/benchmarks/clip_gp_N4_rep0

    # Or import in Python/notebook:
    from rkpcn_analysis.benchmark import run_benchmark, load_benchmark_results

Usage (notebook)::

    from rkpcn_analysis.benchmark import (
        run_benchmark, load_benchmark_results,
        print_benchmark_summary, plot_benchmark_comparison,
    )

    variants = [
        {'label': 'rho99', 'rho': 0.99},
        {'label': 'rho99_u3', 'rho': 0.99, 'n_u_steps': 3},
        {'label': 'rho95', 'rho': 0.95},
    ]
    run_benchmark(rep, variants, output_dir='path/to/output', key=jr.key(42))
    results = load_benchmark_results('path/to/output')
    print_benchmark_summary(results)
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .reconstruct import reconstruct_replicate, load_saved_data
from .runners import run_rkpcn_variant, get_adapted_proposal
from .diagnostics import summary_table, w2_table, integrated_autocorrelation_time
from uncprop.utils.diagnostics import compute_ess
from uncprop.utils.grid import Grid, normalize_density_over_grid


# =============================================================================
# Core benchmark runner
# =============================================================================

def run_benchmark(
    rep,
    variants: list[dict],
    output_dir: str | Path,
    key=None,
    prop_cov=None,
    common_kwargs: dict | None = None,
    save_full_trace: bool = False,
):
    """Run all RKPCN variants and save results to disk.

    For each variant, runs ``run_rkpcn_variant`` and saves:
    - ``<label>/summary.json``: scalar diagnostics (ESS, accept rate, IAT, runtime)
    - ``<label>/samples.npz``: post-burnin samples
    - ``<label>/trace.npz``: full trace (if save_full_trace=True)

    Also saves ``benchmark_meta.json`` with the variant specs and run info.

    Args:
        rep: VSEMReplicate (reconstructed).
        variants: List of dicts, each with at least 'label' and 'rho'.
            Supported keys: label, rho, n_u_steps, prop_cov, n_total,
            n_burnin, initial_position. Unknown keys are ignored.
        output_dir: Where to save results.
        key: PRNG key. If None, uses jr.key(0).
        prop_cov: Default proposal covariance. Individual variants can
            override this.
        common_kwargs: Default kwargs applied to all variants (overridden
            by per-variant settings).
        save_full_trace: If True, save full (unthinned) trace per variant.
            This can be large — set False for production runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if key is None:
        key = jr.key(0)
    if common_kwargs is None:
        common_kwargs = {}

    # Get default proposal if not provided
    if prop_cov is None:
        print('Computing adapted proposal covariance...')
        key, key_adapt = jr.split(key)
        prop_cov, _ = get_adapted_proposal(key_adapt, rep)
        print(f'  prop_cov diag: {np.diag(np.array(prop_cov))}')

    # Save benchmark metadata
    meta = {
        'n_variants': len(variants),
        'variants': variants,
        'common_kwargs': {k: str(v) for k, v in common_kwargs.items()},
        'save_full_trace': save_full_trace,
        'prop_cov_diag': np.diag(np.array(prop_cov)).tolist(),
    }
    with open(output_dir / 'benchmark_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Run each variant
    all_results = {}
    for i, variant in enumerate(variants):
        jax.clear_caches()
        key, subkey = jr.split(key)

        # Merge common kwargs with per-variant overrides
        kwargs = dict(common_kwargs)
        kwargs.update(variant)

        label = kwargs.pop('label', f'variant_{i}')
        v_prop_cov = kwargs.pop('prop_cov', prop_cov)

        # Handle prop_cov_scale: shorthand for scaling the default prop_cov
        if 'prop_cov_scale' in kwargs:
            scale = kwargs.pop('prop_cov_scale')
            v_prop_cov = scale**2 * prop_cov

        print(f'\n[{i+1}/{len(variants)}] {label}')

        result = run_rkpcn_variant(
            key=subkey, rep=rep, prop_cov=v_prop_cov, label=label, **kwargs)
        all_results[label] = result

        # Save per-variant output
        var_dir = output_dir / label
        var_dir.mkdir(exist_ok=True)

        # Samples (post-burnin only, thinned for storage)
        thin = max(1, result['post_burnin'].shape[0] // 2000)
        np.savez(var_dir / 'samples.npz',
                 post_burnin=result['post_burnin'][::thin])

        # Full trace (optional)
        if save_full_trace:
            np.savez(var_dir / 'trace.npz',
                     positions=result['positions'],
                     logdensities=result['logdensities'],
                     accept_probs=result['accept_probs'])

        # Summary stats
        ld_post = result['logdensities'][result['n_burnin']:]
        iat_ld = integrated_autocorrelation_time(ld_post)
        summary = {
            'label': label,
            'rho': result['rho'],
            'n_u_steps': result.get('n_u_steps', 1),
            'accept_rate': result['accept_rate'],
            'min_ess': float(min(result['ess'])),
            'ess': [float(e) for e in result['ess']],
            'iat_logdensity': float(iat_ld),
            'n_post_burnin': result['post_burnin'].shape[0],
            'n_burnin': result['n_burnin'],
            'runtime': result.get('runtime', 0.0),
            'prop_cov_diag': np.diag(np.array(v_prop_cov)).tolist(),
        }
        with open(var_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print(f'\nBenchmark complete. Results saved to {output_dir}')
    return all_results


# =============================================================================
# Loading and analysis
# =============================================================================

def load_benchmark_results(output_dir: str | Path) -> dict:
    """Load all variant results from a benchmark output directory.

    Returns:
        dict mapping label -> {'summary': dict, 'post_burnin': array,
        'trace': dict or None}
    """
    output_dir = Path(output_dir)
    results = {}

    for var_dir in sorted(output_dir.iterdir()):
        if not var_dir.is_dir():
            continue

        summary_path = var_dir / 'summary.json'
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        label = summary['label']

        # Load samples
        samples_path = var_dir / 'samples.npz'
        post_burnin = None
        if samples_path.exists():
            data = np.load(samples_path)
            post_burnin = data['post_burnin']

        # Load trace if available
        trace = None
        trace_path = var_dir / 'trace.npz'
        if trace_path.exists():
            trace_data = np.load(trace_path)
            trace = {k: trace_data[k] for k in trace_data.files}

        results[label] = {
            'summary': summary,
            'post_burnin': post_burnin,
            'trace': trace,
        }

    return results


def load_benchmark_meta(output_dir: str | Path) -> dict:
    """Load benchmark metadata."""
    with open(Path(output_dir) / 'benchmark_meta.json') as f:
        return json.load(f)


def print_benchmark_summary(results: dict, par_names: list[str] | None = None):
    """Print a formatted summary table of benchmark results.

    Args:
        results: dict from load_benchmark_results.
        par_names: parameter names for ESS columns.
    """
    if not results:
        print('No results to summarize.')
        return

    # Determine dimensionality
    first = next(iter(results.values()))
    d = len(first['summary']['ess'])
    if par_names is None:
        par_names = [f'u{i+1}' for i in range(d)]

    ess_cols = ''.join(f' {f"ESS({p})":>10s}' for p in par_names)
    header = (f'{"label":>20s} | {"rho":>5s} | {"n_u":>4s} | '
              f'{"accept":>7s} | {"min ESS":>8s} |{ess_cols} | '
              f'{"IAT(ld)":>8s} | {"time(s)":>8s}')
    print(header)
    print('-' * len(header))

    for label, data in results.items():
        s = data['summary']
        ess_str = ''.join(f' {e:10.1f}' for e in s['ess'])
        print(f'{label:>20s} | {s["rho"]:5.2f} | {s.get("n_u_steps",1):4d} | '
              f'{s["accept_rate"]:7.4f} | {s["min_ess"]:8.1f} |{ess_str} | '
              f'{s["iat_logdensity"]:8.1f} | {s.get("runtime",0):8.1f}')


def compute_benchmark_w2(
    results: dict,
    ep_density,
    grid,
    thin: int = 5,
    reference_samples: dict | None = None,
):
    """Compute W2 distances from each variant to the grid-based EP.

    Uses KDE-on-grid approach from diagnostics module.

    Args:
        results: dict from load_benchmark_results.
        ep_density: (n_grid,) log EP density on grid.
        grid: Grid object.
        thin: thinning factor for samples before KDE.
        reference_samples: optional {name: (n, d) array} for
            additional comparisons (e.g., exact, mean, eup).

    Returns:
        dict mapping label -> W2 distance.
    """
    # Build input dict compatible with w2_table
    w2_input = {}
    if reference_samples is not None:
        for name, samp in reference_samples.items():
            w2_input[name] = {'post_burnin': samp, 'label': name}

    for label, data in results.items():
        if data['post_burnin'] is not None:
            w2_input[label] = {
                'post_burnin': data['post_burnin'],
                'label': label,
            }

    return w2_table(w2_input, ep_grid_density=ep_density, grid=grid, thin=thin)


def plot_benchmark_comparison(
    results: dict,
    ep_density,
    grid,
    reference_samples: dict | None = None,
    thin: int = 5,
    design_points=None,
):
    """Scatter plots of each variant's samples overlaid on EP contours.

    Args:
        results: dict from load_benchmark_results.
        ep_density: (n_grid,) log EP density on grid.
        grid: Grid object.
        reference_samples: optional {name: (n, d) array}.
        thin: thinning for scatter points.
        design_points: (n_design, d) array to overlay on plots.

    Returns:
        (fig, axes)
    """
    from .plots import plot_samples_vs_ep

    # Convert loaded results to the format expected by plot_samples_vs_ep
    plot_results = {}
    for label, data in results.items():
        if data['post_burnin'] is not None:
            plot_results[label] = {
                'post_burnin': data['post_burnin'],
                'label': label,
            }

    fig, axes = plot_samples_vs_ep(
        plot_results, ep_density=ep_density, grid=grid,
        reference_samples=reference_samples, thin=thin)

    # Overlay design points if provided
    if design_points is not None:
        import matplotlib.pyplot as plt
        dp = np.array(design_points)
        for ax_row in np.array(axes).ravel():
            if ax_row.get_visible():
                ax_row.scatter(dp[:, 0], dp[:, 1], c='red', s=80,
                               marker='D', edgecolors='white', zorder=10,
                               label='design')

    return fig, axes


# =============================================================================
# One-call analysis
# =============================================================================

def analyze_benchmark(
    output_dir: str | Path,
    base_dir: str | Path | None = None,
    setup_name: str | None = None,
    rep_idx: int | None = None,
    par_names: list[str] | None = None,
    compute_w2: bool = True,
    save_plots: bool = True,
    thin: int = 5,
):
    """Load benchmark results and produce full analysis.

    Prints summary table, computes W2 (optional), generates scatter
    plots against EP and trace plots. Saves all plots as PDFs.

    This is the main post-processing entry point. Typical usage::

        from rkpcn_analysis.benchmark import analyze_benchmark
        analyze_benchmark('path/to/benchmark_output',
                          base_dir='path/to/experiment',
                          setup_name='clip_gp_N4', rep_idx=0)

    Args:
        output_dir: Benchmark output directory (from run_benchmark).
        base_dir: Experiment base directory (for loading grid densities
            and reference samples). If None, skips EP comparison.
        setup_name: Setup name (e.g., 'clip_gp_N4'). Required if
            base_dir is provided.
        rep_idx: Replicate index. Required if base_dir is provided.
        par_names: Parameter names for display. Defaults to ['u1', 'u2', ...].
        compute_w2: Whether to compute W2 distances (requires base_dir).
        save_plots: Whether to save plots as PDFs to output_dir.
        thin: Thinning for scatter plots and W2 KDE.

    Returns:
        dict with 'results', 'w2' (if computed), 'figures'.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    print('=' * 60)
    print('Benchmark Analysis')
    print(f'  output_dir: {output_dir}')
    print('=' * 60)

    # Load results
    results = load_benchmark_results(output_dir)
    if not results:
        print('No results found.')
        return {'results': {}, 'w2': {}, 'figures': []}

    meta = load_benchmark_meta(output_dir)
    print(f'  {len(results)} variants loaded')

    # Summary table
    print('\n--- Summary ---')
    print_benchmark_summary(results, par_names=par_names)

    # Load experiment data for EP comparison
    ep_density = None
    grid = None
    reference_samples = None
    design_points = None
    figures = []

    if base_dir is not None and setup_name is not None and rep_idx is not None:
        saved = load_saved_data(Path(base_dir), setup_name, rep_idx)

        if saved['grid_densities'] is not None:
            ep_density = saved['grid_densities'].get('ep')

        if saved['grid_info'] is not None:
            gi = saved['grid_info']
            grid = Grid(
                low=np.array(gi['low']),
                high=np.array(gi['high']),
                n_points_per_dim=np.array(gi['n_points_per_dim'], dtype=int),
                dim_names=list(gi['dim_names']) if 'dim_names' in gi else None,
            )
            if par_names is None and 'dim_names' in gi:
                par_names = list(gi['dim_names'])

        if saved['samples'] is not None:
            reference_samples = {}
            for name in ['exact', 'mean', 'eup']:
                if name in saved['samples']:
                    reference_samples[name] = np.array(saved['samples'][name])

        if saved['setup_info'] is not None and 'design_x' in saved['setup_info']:
            design_points = np.array(saved['setup_info']['design_x'])

    # W2 computation
    w2 = {}
    if compute_w2 and ep_density is not None and grid is not None:
        print('\n--- W2 to EP ---')
        w2 = compute_benchmark_w2(
            results, ep_density, grid, thin=thin,
            reference_samples=reference_samples)

    # Scatter plots vs EP
    if ep_density is not None and grid is not None:
        print('\n--- Scatter Plots ---')
        fig, axes = plot_benchmark_comparison(
            results, ep_density, grid,
            reference_samples=reference_samples,
            thin=thin, design_points=design_points)
        figures.append(('scatter_vs_ep', fig))
        if save_plots:
            path = output_dir / 'scatter_vs_ep.pdf'
            fig.savefig(path, bbox_inches='tight', dpi=150)
            print(f'  Saved: {path}')
        plt.close(fig)

        # Also plot reference distributions separately
        from .plots import plot_density_heatmaps
        if saved['grid_densities'] is not None:
            gd = saved['grid_densities']
            available = [n for n in ['exact', 'mean', 'eup', 'ep'] if n in gd]
            fig_hm, _ = plot_density_heatmaps(gd, grid, names=available)
            figures.append(('density_heatmaps', fig_hm))
            if save_plots:
                path = output_dir / 'density_heatmaps.pdf'
                fig_hm.savefig(path, bbox_inches='tight', dpi=150)
                print(f'  Saved: {path}')
            plt.close(fig_hm)

    # Trace plots (for variants with saved traces)
    has_traces = any(d['trace'] is not None for d in results.values())
    if has_traces:
        print('\n--- Trace Plots ---')
        trace_results = {}
        for label, data in results.items():
            if data['trace'] is not None:
                n_burnin = data['summary']['n_burnin']
                trace_results[label] = {
                    'positions': data['trace']['positions'],
                    'logdensities': data['trace']['logdensities'],
                    'post_burnin': data['trace']['positions'][n_burnin:],
                    'n_burnin': n_burnin,
                }

        if trace_results:
            from .plots import plot_traces, plot_acf
            trace_figs = plot_traces(trace_results, par_names=par_names)
            for j, fig in enumerate(trace_figs):
                figures.append((f'traces_{j}', fig))
                if save_plots:
                    path = output_dir / f'traces_{j}.pdf'
                    fig.savefig(path, bbox_inches='tight', dpi=150)
                    print(f'  Saved: {path}')
                plt.close(fig)

            fig_acf, _ = plot_acf(trace_results, par_names=par_names)
            figures.append(('acf', fig_acf))
            if save_plots:
                path = output_dir / 'acf.pdf'
                fig_acf.savefig(path, bbox_inches='tight', dpi=150)
                print(f'  Saved: {path}')
            plt.close(fig_acf)

    print('\n--- Done ---')
    return {'results': results, 'w2': w2, 'figures': figures}


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    """CLI entry point with 'run' and 'analyze' subcommands.

    Usage::

        # Run a benchmark
        python -m rkpcn_analysis.benchmark run \\
            --experiment-name vsem --setup clip_gp_N4 --rep 0 \\
            --config rkpcn_analysis/example_benchmark.yaml \\
            --output-dir ../../out/vsem/benchmarks/clip_gp_N4_rep0

        # Analyze results (after run completes)
        python -m rkpcn_analysis.benchmark analyze \\
            --output-dir ../../out/vsem/benchmarks/clip_gp_N4_rep0 \\
            --experiment-name vsem --setup clip_gp_N4 --rep 0
    """
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    parser = argparse.ArgumentParser(
        description='RKPCN benchmark: run variants or analyze results')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- 'run' subcommand ---
    run_parser = subparsers.add_parser('run', help='Run benchmark variants')
    run_parser.add_argument('--experiment-name', type=str, required=True)
    run_parser.add_argument('--setup', type=str, required=True,
                            help='Setup name, e.g., clip_gp_N4')
    run_parser.add_argument('--rep', type=int, required=True)
    run_parser.add_argument('--config', type=str, required=True,
                            help='YAML file with variant definitions')
    run_parser.add_argument('--output-dir', type=str, required=True)
    run_parser.add_argument('--seed', type=int, default=42)
    run_parser.add_argument('--save-trace', action='store_true',
                            help='Save full unthinned trace per variant')

    # --- 'analyze' subcommand ---
    analyze_parser = subparsers.add_parser('analyze',
                                            help='Analyze benchmark results')
    analyze_parser.add_argument('--output-dir', type=str, required=True)
    analyze_parser.add_argument('--experiment-name', type=str, default=None,
                                help='For loading EP densities and reference samples')
    analyze_parser.add_argument('--setup', type=str, default=None)
    analyze_parser.add_argument('--rep', type=int, default=None)
    analyze_parser.add_argument('--no-w2', action='store_true',
                                help='Skip W2 computation')

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    if args.command == 'run':
        import yaml

        with open(args.config) as f:
            config = yaml.safe_load(f)
        variants = config['variants']
        common_kwargs = config.get('common_kwargs', {})

        base_dir = repo_root / 'out' / args.experiment_name
        rep, _ = reconstruct_replicate(base_dir, args.setup, args.rep)

        key = jr.key(args.seed)
        run_benchmark(
            rep=rep,
            variants=variants,
            output_dir=args.output_dir,
            key=key,
            common_kwargs=common_kwargs,
            save_full_trace=args.save_trace,
        )

    elif args.command == 'analyze':
        base_dir = None
        if args.experiment_name is not None:
            base_dir = repo_root / 'out' / args.experiment_name

        analyze_benchmark(
            output_dir=args.output_dir,
            base_dir=base_dir,
            setup_name=args.setup,
            rep_idx=args.rep,
            compute_w2=not args.no_w2,
        )


if __name__ == '__main__':
    main()
