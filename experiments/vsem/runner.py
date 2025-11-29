# experiments/vsem/runner.py
import numpy as np
import matplotlib.pyplot as plt

import vsem_jax as vsem
from vsem_inv_prob import (
    InvProb, 
    VSEMPrior, 
    VSEMLikelihood,
    VSEMTest
)


def run_vsem_experiment(rng: np.random.Generator,
                        n_design: int,
                        n_test_grid_1d: int, 
                        n_reps: int) -> tuple[list[VSEMTest], list, list]:
    """ Run VSEM experiment for paper

    The structure of the inverse problem and surrogate model are fixed
    across replications. Replications average over the randomness in the 
    inverse problem setup (sampling synthetic data, ground truth value, etc.)
    and the sampling of the design points. The number of design points is 
    fixed across replications.

    Returns:
        tuple, containing:
            list of VSEMTest objects over the replications
            list of evaluation metrics over the replications
            list of iteration indices that failed
    """
    
    test_list = []
    metrics_list = []
    failed_iters = []

    for i in range(n_reps):
        print(f'Replication {i+1}')

        try:
            inv_prob = get_vsem_inv_prob(rng)
            vsem_test = VSEMTest(inv_prob, n_design=n_design, n_test_grid_1d=n_test_grid_1d)
            metrics = vsem_test.compute_metrics(pred_type='pred')
            test_list.append(vsem_test)
            metrics_list.append(metrics)
        except Exception as e:
            print(f'Iteration {i} failed with error: {e}')
            failed_iters.append(i)
            test_list.append(e)
            metrics_list.append(e)

    print(f'Number of failed iterations: {len(failed_iters)}')
    return test_list, metrics_list, failed_iters


def get_vsem_inv_prob(rng: np.random.Generator) -> InvProb:
    n_days = 365 * 2
    par_names = ["kext", "av"]

    # For exact MCMC
    proposal_cov = np.diag([0.1**2, 0.04**2])

    prior = VSEMPrior(par_names, rng)
    ground_truth = prior.simulate_ground_truth()
    likelihood = VSEMLikelihood(rng, n_days, par_names, ground_truth=ground_truth)
    inv_prob = InvProb(rng, prior, likelihood, proposal_cov=proposal_cov)

    return inv_prob


def plot_coverage(tests: list[VSEMTest], 
                  metrics: list, 
                  q_min: float = 0.05, 
                  q_max: float = 0.95, 
                  figsize=(12, 4)):
    """
    The first two arguments are those returned by `run_vsem_experiment()`.
    Assumes the same coverage probabilities were used for all replications
    within the experiment.
    """

    # Assumed constrant across all replications
    probs = metrics[0]['alphas']

    n_reps = len(tests)
    n_probs = len(probs)
    mean_coverage = np.empty((n_reps, n_probs))
    eup_coverage = np.empty((n_reps, n_probs))
    ep_coverage = np.empty((n_reps, n_probs))

    # assemble arrays of coverage stats
    for i, results in enumerate(metrics):
        mean, eup, ep = results['coverage']
        mean_coverage[i,:] = mean
        eup_coverage[i,:] = eup
        ep_coverage[i,:] = ep

    # summarize distribution over replications
    mean_m = np.median(mean_coverage, axis=0)
    eup_m = np.median(eup_coverage, axis=0)
    ep_m = np.median(ep_coverage, axis=0)
    mean_q = np.quantile(mean_coverage, q=[q_min, q_max], axis=0)
    eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)
    ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)

    meds = [mean_m, eup_m, ep_m]
    qs = [mean_q, eup_q, ep_q]
    labels = ['mean', 'eup', 'ep']
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs = axs.reshape(-1)
    n_plots = len(axs)

    for i in range(n_plots):
        ax = axs[i]
        q = qs[i]
        med = meds[i]
        label = labels[i]

        ax.fill_between(probs, q[0,:], q[1,:], alpha=0.7)
        ax.plot(probs, med)
        ax.set_title(label)
        ax.set_xlabel('Nominal Coverage')
        ax.set_ylabel('Actual Coverage')

        # Add line y = x
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        y = x
        ax.plot(x, y, color="red", linestyle="--")
        ax.legend()

    plt.close(fig)
    return fig, axs


def plot_coverage_same_plot(tests: list[VSEMTest], 
                            metrics: list, 
                            q_min: float = 0.05, 
                            q_max: float = 0.95, 
                            ax=None):
    """
    The first two arguments are those returned by `run_vsem_experiment()`.
    Assumes the same coverage probabilities were used for all replications
    within the experiment.
    """

    # Assumed constrant across all replications
    probs = metrics[0]['alphas']

    n_reps = len(tests)
    n_probs = len(probs)
    mean_coverage = np.empty((n_reps, n_probs))
    eup_coverage = np.empty((n_reps, n_probs))
    ep_coverage = np.empty((n_reps, n_probs))

    # assemble arrays of coverage stats
    for i, results in enumerate(metrics):
        mean, eup, ep = results['coverage']
        mean_coverage[i,:] = mean
        eup_coverage[i,:] = eup
        ep_coverage[i,:] = ep

    # summarize distribution over replications
    mean_m = np.median(mean_coverage, axis=0)
    eup_m = np.median(eup_coverage, axis=0)
    ep_m = np.median(ep_coverage, axis=0)
    mean_q = np.quantile(mean_coverage, q=[q_min, q_max], axis=0)
    eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)
    ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.fill_between(probs, mean_q[0,:], mean_q[1,:], color='green', alpha=0.3, label='mean')
    ax.fill_between(probs, eup_q[0,:], eup_q[1,:], color='red', alpha=0.3, label='eup')
    ax.fill_between(probs, ep_q[0,:], ep_q[1,:], color='blue', alpha=0.3, label='ep')
    ax.plot(probs, mean_m, color='green', label='mean')
    ax.plot(probs, eup_m, color='red', label='eup')
    ax.plot(probs, ep_m, color='blue', label='ep')
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color="red", linestyle="--")
    ax.legend()
    plt.close(fig)

    return fig, ax


def plot_coverage_single_rep(probs, coverage_list, *, labels=None, figsize=(5,4)):
    fig, ax = plt.subplots(figsize=figsize)
    n_curves = len(coverage_list)
    if labels is None:
        labels = [f"Plot {i}" for i in range(n_curves)]

    for j in range(n_curves):
        ax.plot(probs, coverage_list[j], label=labels[j])

    ax.set_title("Coverage")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color="red", linestyle="--")
    ax.legend()

    # Close figure
    plt.close(fig)
    return fig