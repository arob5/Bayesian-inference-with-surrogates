# experiments/vsem/runner.py
import numpy as np
import matplotlib.pyplot as plt

import vsem_jax as vsem
from inverse_problem import InvProb, VSEMPrior, VSEMLikelihood
from surrogate import VSEMTest


def run_vsem_experiment(rng: np.random.Generator,
                        n_design: int,
                        design_method: str,
                        n_test_grid_1d: int, 
                        n_reps: int,
                        store_pred_clipped: bool = True) -> tuple[list[VSEMTest], list, list, list]:
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
    metrics_clip_list = []
    failed_iters = []

    for i in range(n_reps):
        print(f'Replication {i+1}')

        try:
            inv_prob = get_vsem_inv_prob(rng)
            vsem_test = VSEMTest(inv_prob=inv_prob, 
                                 n_design=n_design, 
                                 n_test_grid_1d=n_test_grid_1d,
                                 store_pred_clipped=store_pred_clipped,
                                 design_method=design_method)
            
            metrics = vsem_test.compute_metrics(pred_type='pred')
            metrics_list.append(metrics)
            test_list.append(vsem_test)

            if store_pred_clipped:
                metrics_clip = vsem_test.compute_metrics(pred_type='pred_clip')
                metrics_clip_list.append(metrics_clip)
        except Exception as e:
            print(f'Iteration {i} failed with error: {e}')
            failed_iters.append(i)
            test_list.append(e)
            metrics_list.append(e)
            if store_pred_clipped:
                metrics_clip_list.append(e)

    print(f'Number of failed iterations: {len(failed_iters)}')
    return test_list, metrics_list, metrics_clip_list, failed_iters


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