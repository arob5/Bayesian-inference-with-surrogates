"""
EP baseline validation for PDE experiment.

Validates the MCwMH expected posterior approximation for specific replicates.
Produces appendix/supplementary figures. Run separately from the main experiment.

Three studies:
  1. MCwMH convergence: compare standard vs. heavy budget
  2. RFF convergence: vary num_rff and compare EP samples
  3. Chain quality: analyze per-chain diagnostics, identify outlier chains

Usage:
    # Run all studies for a specific replicate:
    python validate_ep.py --experiment-name pde_local_test2 \
        --n-design 4 --rep 0 --output-dir ../../out/pde_local_test2/validation

    # Chain quality only (fast, no MCMC re-runs):
    python validate_ep.py --experiment-name pde_local_test2 \
        --n-design 4 --rep 0 --output-dir ../../out/pde_local_test2/validation \
        --studies chain_quality

    # Submit as batch job for heavy MCwMH:
    qsub submit_validate.sh
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['XLA_FLAGS'] = (
    '--xla_cpu_multi_thread_eigen=false '
    'intra_op_parallelism_threads=1 '
    'inter_op_parallelism_threads=1'
)

from jax import config
config.update('jax_enable_x64', True)

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment import PDEReplicate, sample_mcwmh
from uncprop.utils.diagnostics import compute_ess
from uncprop.utils.wasserstein import wasserstein2_sinkhorn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rep_dir(base_dir, n_design, rep_idx):
    return Path(base_dir) / f'n_design_{n_design}' / f'rep{rep_idx}'


def load_rep(base_dir, n_design, rep_idx, num_rff_override=None):
    """Reconstruct a PDEReplicate from saved keys.

    Args:
        num_rff_override: if provided, re-initialize with a different num_rff
            (same GP fit, different RFF basis). Used for RFF convergence study.
    """
    rep_out_dir = _rep_dir(base_dir, n_design, rep_idx)
    init_settings = jnp.load(rep_out_dir / 'init_settings.npz')
    key_init = jr.wrap_key_data(init_settings['key_init'])

    num_rff = num_rff_override if num_rff_override else int(init_settings['num_rff'].item())

    rep = PDEReplicate(
        key=key_init,
        out_dir=rep_out_dir,
        n_design=n_design,
        num_rff=num_rff,
        design_method=init_settings['design_method'].item(),
        write_to_file=False,
    )
    return rep


def _pair_labels(dim):
    """Generate parameter pair labels for 2D marginal plots."""
    pairs = []
    for i in range(dim):
        for j in range(i + 1, dim):
            pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Study 1: MCwMH convergence
# ---------------------------------------------------------------------------

def study_mcwmh_convergence(base_dir, n_design, rep_idx, output_dir,
                             standard_settings=None, heavy_settings=None,
                             seed=42):
    """Compare standard vs. heavy MCwMH budget for a given replicate.

    Reconstructs the replicate, runs MCwMH at two budgets, and compares
    the resulting EP approximations.
    """
    print('\n' + '='*60)
    print('Study 1: MCwMH Convergence')
    print('='*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jr.key(seed)

    if standard_settings is None:
        standard_settings = {
            'n_chains': 200, 'n_samp_per_chain': 50,
            'n_burnin': 10_000, 'thin_window': 100,
        }
    if heavy_settings is None:
        heavy_settings = {
            'n_chains': 500, 'n_samp_per_chain': 200,
            'n_burnin': 10_000, 'thin_window': 100,
        }

    # Reconstruct replicate
    print(f'\nReconstructing replicate (n_design={n_design}, rep={rep_idx})...')
    start = time.perf_counter()
    rep = load_rep(base_dir, n_design, rep_idx)
    print(f'  Reconstruction time: {time.perf_counter() - start:.1f}s')

    # Get adapted proposal covariance from existing samples
    rep_out_dir = _rep_dir(base_dir, n_design, rep_idx)
    existing_samp = dict(jnp.load(rep_out_dir / 'samples.npz'))
    if 'mean' in existing_samp:
        prop_cov = jnp.cov(existing_samp['mean'], rowvar=False)
    else:
        prop_cov = 0.3**2 * jnp.eye(rep.posterior.dim)

    # Run standard budget
    key, key_std = jr.split(key)
    print(f'\nRunning standard MCwMH ({standard_settings["n_chains"]} chains '
          f'x {standard_settings["n_samp_per_chain"]} samples)...')
    start = time.perf_counter()
    std_result = sample_mcwmh(
        key=key_std,
        posterior_surrogate=rep.posterior_surrogate,
        prop_cov_init=prop_cov,
        **standard_settings,
    )
    std_time = time.perf_counter() - start
    std_samp = std_result['samples'].reshape(-1, rep.posterior.dim)
    print(f'  Time: {std_time:.1f}s')
    print(f'  Median accept rate: {np.median(std_result["accept_rates"]):.4f}')
    print(f'  Median ESS: {np.median(std_result["ess"]):.1f}')

    # Run heavy budget
    key, key_heavy = jr.split(key)
    print(f'\nRunning heavy MCwMH ({heavy_settings["n_chains"]} chains '
          f'x {heavy_settings["n_samp_per_chain"]} samples)...')
    start = time.perf_counter()
    heavy_result = sample_mcwmh(
        key=key_heavy,
        posterior_surrogate=rep.posterior_surrogate,
        prop_cov_init=prop_cov,
        **heavy_settings,
    )
    heavy_time = time.perf_counter() - start
    heavy_samp = heavy_result['samples'].reshape(-1, rep.posterior.dim)
    print(f'  Time: {heavy_time:.1f}s')
    print(f'  Median accept rate: {np.median(heavy_result["accept_rates"]):.4f}')
    print(f'  Median ESS: {np.median(heavy_result["ess"]):.1f}')

    # W2 between standard and heavy
    key, key_w2 = jr.split(key)
    w2_std_heavy = wasserstein2_sinkhorn(
        std_samp, heavy_samp,
        threshold=1e-6, max_iterations=5000,
    )
    print(f'\nW2(standard, heavy) = {w2_std_heavy:.6f}')

    # Context: W2 from EP to other approximations (if available)
    for ref_name in ['eup', 'mean', 'exact']:
        if ref_name in existing_samp:
            key, key_ctx = jr.split(key)
            w2_ctx = wasserstein2_sinkhorn(
                heavy_samp, existing_samp[ref_name],
                threshold=1e-6, max_iterations=5000,
            )
            print(f'W2(heavy_EP, {ref_name}) = {w2_ctx:.6f}')

    # Save results
    jnp.savez(output_dir / f'mcwmh_convergence_n{n_design}_rep{rep_idx}.npz',
              std_samples=std_samp,
              heavy_samples=heavy_samp,
              std_accept_rates=std_result['accept_rates'],
              heavy_accept_rates=heavy_result['accept_rates'],
              std_ess=std_result['ess'],
              heavy_ess=heavy_result['ess'],
              w2_std_heavy=w2_std_heavy)

    # 2D marginal plots
    dim = rep.posterior.dim
    pairs = _pair_labels(dim)[:6]  # up to 6 pairs
    n_pairs = len(pairs)
    fig, axs = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), squeeze=False)

    for k, (i, j) in enumerate(pairs):
        ax = axs[0, k]
        ax.scatter(heavy_samp[:, i], heavy_samp[:, j],
                   alpha=0.15, s=5, label='heavy', color='C0')
        ax.scatter(std_samp[:, i], std_samp[:, j],
                   alpha=0.15, s=5, label='standard', color='C1')
        ax.set_xlabel(f'u{i+1}')
        ax.set_ylabel(f'u{j+1}')
        if k == 0:
            ax.legend(markerscale=3)

    fig.suptitle(f'MCwMH convergence (n_design={n_design}, rep={rep_idx})\n'
                 f'W2(std, heavy) = {w2_std_heavy:.4f}', fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f'mcwmh_convergence_n{n_design}_rep{rep_idx}.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots to {output_dir}')

    return {'std_result': std_result, 'heavy_result': heavy_result,
            'w2_std_heavy': w2_std_heavy}


# ---------------------------------------------------------------------------
# Study 2: RFF convergence
# ---------------------------------------------------------------------------

def study_rff_convergence(base_dir, n_design, rep_idx, output_dir,
                           num_rff_values=None, mcwmh_settings=None,
                           seed=43):
    """Vary num_rff and compare resulting EP samples.

    Re-initializes the replicate with different num_rff values (same GP fit,
    different RFF basis), runs MCwMH with each, and compares via W2.
    """
    print('\n' + '='*60)
    print('Study 2: RFF Convergence')
    print('='*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jr.key(seed)

    if num_rff_values is None:
        num_rff_values = [500, 1000, 2000]
    if mcwmh_settings is None:
        mcwmh_settings = {
            'n_chains': 200, 'n_samp_per_chain': 50,
            'n_burnin': 10_000, 'thin_window': 100,
        }

    # Get adapted proposal covariance
    rep_out_dir = _rep_dir(base_dir, n_design, rep_idx)
    existing_samp = dict(jnp.load(rep_out_dir / 'samples.npz'))
    dim = existing_samp['exact'].shape[1] if 'exact' in existing_samp else 6

    if 'mean' in existing_samp:
        prop_cov = jnp.cov(existing_samp['mean'], rowvar=False)
    else:
        prop_cov = 0.3**2 * jnp.eye(dim)

    ep_samples = {}

    for num_rff in num_rff_values:
        print(f'\nnum_rff = {num_rff}')
        print(f'  Reconstructing replicate...')
        start = time.perf_counter()
        rep = load_rep(base_dir, n_design, rep_idx, num_rff_override=num_rff)
        print(f'  Reconstruction time: {time.perf_counter() - start:.1f}s')

        key, key_mcwmh = jr.split(key)
        print(f'  Running MCwMH ({mcwmh_settings["n_chains"]} chains '
              f'x {mcwmh_settings["n_samp_per_chain"]} samples)...')
        start = time.perf_counter()
        result = sample_mcwmh(
            key=key_mcwmh,
            posterior_surrogate=rep.posterior_surrogate,
            prop_cov_init=prop_cov,
            **mcwmh_settings,
        )
        elapsed = time.perf_counter() - start
        samp = result['samples'].reshape(-1, rep.posterior.dim)
        ep_samples[num_rff] = samp
        print(f'  Time: {elapsed:.1f}s, median accept: '
              f'{np.median(result["accept_rates"]):.4f}')

        jax.clear_caches()

    # Pairwise W2
    rff_vals = sorted(ep_samples.keys())
    print('\nPairwise W2:')
    w2_pairs = {}
    for i, a in enumerate(rff_vals):
        for b in rff_vals[i+1:]:
            key, key_w2 = jr.split(key)
            w2 = wasserstein2_sinkhorn(
                ep_samples[a], ep_samples[b],
                threshold=1e-6, max_iterations=5000,
            )
            w2_pairs[(a, b)] = float(w2)
            print(f'  W2(rff={a}, rff={b}) = {w2:.6f}')

    # Save
    save_dict = {f'ep_rff{r}': ep_samples[r] for r in rff_vals}
    save_dict['num_rff_values'] = np.array(rff_vals)
    for (a, b), v in w2_pairs.items():
        save_dict[f'w2_rff{a}_rff{b}'] = np.array(v)
    jnp.savez(output_dir / f'rff_convergence_n{n_design}_rep{rep_idx}.npz',
              **save_dict)

    # Plot: 2D marginals for each RFF value
    pairs = _pair_labels(dim)[:4]
    n_pairs = len(pairs)
    fig, axs = plt.subplots(len(rff_vals), n_pairs,
                            figsize=(4 * n_pairs, 4 * len(rff_vals)),
                            squeeze=False)

    for row, num_rff in enumerate(rff_vals):
        samp = np.array(ep_samples[num_rff])
        for col, (i, j) in enumerate(pairs):
            ax = axs[row, col]
            ax.scatter(samp[:, i], samp[:, j], alpha=0.1, s=3, color='C0')
            ax.set_xlabel(f'u{i+1}')
            ax.set_ylabel(f'u{j+1}')
            if col == 0:
                ax.set_ylabel(f'rff={num_rff}\nu{j+1}')

    fig.suptitle(f'RFF convergence (n_design={n_design}, rep={rep_idx})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f'rff_convergence_n{n_design}_rep{rep_idx}.pdf',
                bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plots to {output_dir}')

    return ep_samples, w2_pairs


# ---------------------------------------------------------------------------
# Study 3: Chain quality analysis
# ---------------------------------------------------------------------------

def study_chain_quality(base_dir, n_design, rep_idx, output_dir,
                         ess_threshold_quantile=0.05,
                         seed=44):
    """Analyze per-chain diagnostics and identify outlier chains.

    Uses saved diagnostics.npz (no MCMC re-runs needed).
    """
    print('\n' + '='*60)
    print('Study 3: Chain Quality Analysis')
    print('='*60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = jr.key(seed)

    rep_out_dir = _rep_dir(base_dir, n_design, rep_idx)
    diag = dict(jnp.load(rep_out_dir / 'diagnostics.npz'))
    samp = dict(jnp.load(rep_out_dir / 'samples.npz'))

    if 'mcwmh_accept_rates' not in diag:
        print('  No per-chain MCwMH diagnostics found. '
              'Re-run the experiment with updated code.')
        return None

    accept_rates = np.array(diag['mcwmh_accept_rates'])
    ess_per_chain = np.array(diag['mcwmh_ess'])
    final_logdens = np.array(diag['mcwmh_final_logdens'])

    n_chains = len(accept_rates)
    dim = ess_per_chain.shape[1]
    min_ess = ess_per_chain.min(axis=1)  # (n_chains,)

    print(f'\n  n_chains: {n_chains}')
    print(f'  Accept rates: median={np.median(accept_rates):.4f}, '
          f'min={np.min(accept_rates):.4f}, max={np.max(accept_rates):.4f}')
    print(f'  Min ESS: median={np.median(min_ess):.1f}, '
          f'min={np.min(min_ess):.1f}, max={np.max(min_ess):.1f}')
    print(f'  Final log-density: median={np.median(final_logdens):.2f}, '
          f'min={np.min(final_logdens):.2f}, max={np.max(final_logdens):.2f}')

    # Identify outlier chains
    ess_threshold = np.quantile(min_ess, ess_threshold_quantile)
    outlier_mask = min_ess < ess_threshold
    n_outliers = int(np.sum(outlier_mask))
    print(f'\n  ESS threshold ({ess_threshold_quantile:.0%} quantile): {ess_threshold:.1f}')
    print(f'  Outlier chains: {n_outliers}/{n_chains}')

    # EP samples with/without outlier chains
    if 'ep_mcwmh' not in samp:
        print('  No ep_mcwmh samples found.')
        return None

    ep_samp_full = np.array(samp['ep_mcwmh'])
    n_per_chain = ep_samp_full.shape[0] // n_chains

    # Reshape to (n_chains, n_per_chain, dim) then filter
    ep_by_chain = ep_samp_full.reshape(n_chains, n_per_chain, dim)
    good_mask = ~outlier_mask
    ep_samp_filtered = ep_by_chain[good_mask].reshape(-1, dim)

    print(f'  Full EP samples: {ep_samp_full.shape[0]}')
    print(f'  Filtered EP samples: {ep_samp_filtered.shape[0]}')

    # W2 between full and filtered
    if n_outliers > 0 and n_outliers < n_chains:
        key, key_w2 = jr.split(key)
        w2_filter = wasserstein2_sinkhorn(
            jnp.array(ep_samp_full), jnp.array(ep_samp_filtered),
            threshold=1e-6, max_iterations=5000,
        )
        print(f'  W2(full, filtered) = {float(w2_filter):.6f}')
    else:
        w2_filter = 0.0

    # --- Plots ---

    # Fig 1: Histograms of per-chain diagnostics
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].hist(accept_rates, bins=30, edgecolor='black', linewidth=0.5)
    axs[0].set_xlabel('Accept rate')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Per-chain accept rates')

    axs[1].hist(min_ess, bins=30, edgecolor='black', linewidth=0.5)
    axs[1].axvline(ess_threshold, color='red', linestyle='--',
                    label=f'{ess_threshold_quantile:.0%} quantile')
    axs[1].set_xlabel('Min ESS (across params)')
    axs[1].set_title('Per-chain min ESS')
    axs[1].legend()

    axs[2].hist(final_logdens, bins=30, edgecolor='black', linewidth=0.5)
    axs[2].set_xlabel('Final log-density')
    axs[2].set_title('Per-chain final log-density')

    fig.suptitle(f'Chain diagnostics (n_design={n_design}, rep={rep_idx})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f'chain_diagnostics_n{n_design}_rep{rep_idx}.pdf',
                bbox_inches='tight')
    plt.close(fig)

    # Fig 2: EP samples colored by chain index
    pairs = _pair_labels(dim)[:4]
    n_pairs = len(pairs)
    fig, axs = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), squeeze=False)

    # Color by chain: use a colormap
    chain_idx = np.repeat(np.arange(n_chains), n_per_chain)
    for k, (i, j) in enumerate(pairs):
        ax = axs[0, k]
        sc = ax.scatter(ep_samp_full[:, i], ep_samp_full[:, j],
                        c=chain_idx, cmap='tab20', alpha=0.3, s=3)
        ax.set_xlabel(f'u{i+1}')
        ax.set_ylabel(f'u{j+1}')

    fig.suptitle(f'EP samples colored by chain (n_design={n_design}, rep={rep_idx})',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f'chain_colored_n{n_design}_rep{rep_idx}.pdf',
                bbox_inches='tight')
    plt.close(fig)

    # Fig 3: Full vs filtered EP comparison
    if n_outliers > 0:
        fig, axs = plt.subplots(1, n_pairs, figsize=(4 * n_pairs, 4), squeeze=False)
        for k, (i, j) in enumerate(pairs):
            ax = axs[0, k]
            ax.scatter(ep_samp_full[:, i], ep_samp_full[:, j],
                       alpha=0.1, s=3, color='C3', label='full')
            ax.scatter(ep_samp_filtered[:, i], ep_samp_filtered[:, j],
                       alpha=0.1, s=3, color='C0', label='filtered')
            ax.set_xlabel(f'u{i+1}')
            ax.set_ylabel(f'u{j+1}')
            if k == 0:
                ax.legend(markerscale=3)

        fig.suptitle(f'Full vs filtered EP (W2={float(w2_filter):.4f}, '
                     f'{n_outliers} chains removed)', fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f'chain_filtered_n{n_design}_rep{rep_idx}.pdf',
                    bbox_inches='tight')
        plt.close(fig)

    print(f'Saved plots to {output_dir}')

    return {
        'accept_rates': accept_rates,
        'ess_per_chain': ess_per_chain,
        'final_logdens': final_logdens,
        'outlier_mask': outlier_mask,
        'w2_filter': w2_filter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STUDY_NAMES = ['mcwmh_convergence', 'rff_convergence', 'chain_quality']


def main():
    parser = argparse.ArgumentParser(
        description='Validate the MCwMH EP baseline for specific PDE replicates')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--n-design', type=int, required=True)
    parser.add_argument('--rep', type=int, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base output directory (default: ../../out/<experiment-name>)')
    parser.add_argument('--studies', nargs='+', default=STUDY_NAMES,
                        choices=STUDY_NAMES,
                        help='Which studies to run (default: all)')
    # MCwMH convergence settings
    parser.add_argument('--heavy-n-chains', type=int, default=500)
    parser.add_argument('--heavy-n-samp', type=int, default=200)
    # RFF settings
    parser.add_argument('--num-rff-values', nargs='+', type=int,
                        default=[500, 1000, 2000])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.base_dir is None:
        base_dir = repo_root / 'out' / args.experiment_name
    else:
        base_dir = Path(args.base_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('PDE EP Baseline Validation')
    print(f'  experiment: {args.experiment_name}')
    print(f'  n_design: {args.n_design}, rep: {args.rep}')
    print(f'  studies: {args.studies}')
    print(f'  output: {output_dir}')
    print('='*60)

    if 'chain_quality' in args.studies:
        study_chain_quality(
            base_dir, args.n_design, args.rep, output_dir, seed=args.seed)

    if 'mcwmh_convergence' in args.studies:
        heavy_settings = {
            'n_chains': args.heavy_n_chains,
            'n_samp_per_chain': args.heavy_n_samp,
            'n_burnin': 10_000,
            'thin_window': 100,
        }
        study_mcwmh_convergence(
            base_dir, args.n_design, args.rep, output_dir,
            heavy_settings=heavy_settings, seed=args.seed)

    if 'rff_convergence' in args.studies:
        study_rff_convergence(
            base_dir, args.n_design, args.rep, output_dir,
            num_rff_values=args.num_rff_values, seed=args.seed + 1)

    print('\n' + '='*60)
    print('Validation complete.')
    print('='*60)


if __name__ == '__main__':
    main()
