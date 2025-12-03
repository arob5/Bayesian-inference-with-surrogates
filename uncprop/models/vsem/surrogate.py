# uncprop/models/vsem/surrogate.py
from __future__ import annotations

import jax.numpy as jnp
from gpjax import Dataset
from numpyro.distributions import MultivariateNormal

from uncprop.core.surrogate import Surrogate, PredDist
from uncprop.core.inverse_problem import Posterior, ArrayLike
from uncprop.utils.gpjax_models import construct_gp, train_gp_hyperpars


class VSEMSurrogate(Surrogate):

    def __init__(self, 
                 design: Dataset,
                 exact_posterior: Posterior,
                 gp_train_args: dict | None = None):
        
        if design.in_dim != exact_posterior.dim:
            raise ValueError(f'Design and posterior dim mismatch: {design.in_dim } vs {exact_posterior.dim}')
        if gp_train_args is None:
            gp_train_args = {}
        
        gp_untuned, bijection = construct_gp(design, set_bounds=True)
        gp, opt_info = train_gp_hyperpars(gp_untuned, bijection, design, **gp_train_args)

        self.design = design
        self.gp = gp
        self._gp_untuned = gp_untuned
        self._opt_info = opt_info
        self._input_dim = exact_posterior.dim

    def __call__(self, input: ArrayLike) -> PredDist:
        """ Convert gpjax Gaussian to a pure numpyro Gaussian """
        pred_latent = self.gp(input, train_data=self.design)
        pred = self.gp.likelihood(pred_latent)
        return pred

    @property
    def input_dim(self) -> int:
        return self._input_dim
    
    def summarize_fit(self):
        start_loss = self._opt_info['starting_loss']
        end_loss = self._opt_info['final_loss']
        gp_scale = jnp.sqrt(self.gp.prior.kernel.variance.get_value())
        gp_ls = self.gp.prior.kernel.lengthscale.get_value()
        gp_noise_sd = self.gp.likelihood.obs_stddev.get_value()

        print(f'Initial loss {start_loss}')
        print(f'Final loss {end_loss}')
        print(f'gp scale: {gp_scale}')
        print(f'gp lengthscales: {gp_ls}')
        print(f'gp noise std dev: {gp_noise_sd}')


class VSEMTest:
    """ Surrogate uncertainty propagation experiment for surrogate modeling for VSEM inverse problem.

    Note that this class is specialized for an inverse problem with a 2d input space.
    """
    
    def __init__(self, inv_prob: InvProb, n_design: int, 
                 n_test_grid_1d: int = 100, store_pred_clipped=True,
                 design_method='lhc'):
        self.inv_prob = inv_prob
        self.n_design = n_design
        self.set_test_grid_info(n_test_grid_1d)
        self.set_gp_model(design_method)
        self.store_gp_pred(store_pred_clipped=store_pred_clipped)
        self.store_posterior_estimates(store_pred_clipped=store_pred_clipped)

    def set_gp_model(self, design_method):
        self.design = self.construct_design(design_method)
        self.gp_prior = self.construct_gp_prior()
        self.gp_likelihood = self.construct_gp_likelihood()
        self.gp_untuned_posterior = self.gp_prior * self.gp_likelihood
        self.gp_posterior, self.opt_info = self.train_gp_hyperpars()

    def set_test_grid_info(self, n_test_grid_1d):
        """
        Note that this method currently assumes 2d bounded input space.
        """
        par_names = self.inv_prob.par_names
        u1_supp = self.inv_prob.prior.dists[par_names[0]].support()
        u2_supp = self.inv_prob.prior.dists[par_names[1]].support()

        u1_grid = np.linspace(u1_supp[0], u1_supp[1], n_test_grid_1d)
        u2_grid = np.linspace(u2_supp[0], u2_supp[1], n_test_grid_1d)
        grid_spacing = (u1_grid[1] - u1_grid[0], u2_grid[1] - u2_grid[0])
        grid_cell_area = grid_spacing[0] * grid_spacing[1]

        U1_grid, U2_grid = np.meshgrid(u1_grid, u2_grid, indexing='xy')
        U_grid= np.stack([U1_grid.ravel(), U2_grid.ravel()], axis=1)
        log_post = self.inv_prob.log_posterior_density(U_grid)
        log_post_grid = log_post.reshape(U1_grid.shape)
        log_post_upper_bound = self.log_post_upper_bound(U_grid)

        self.test_grid_info = {
            "u1_grid": u1_grid,
            "u2_grid": u2_grid,
            "grid_spacing": grid_spacing,
            "cell_area": grid_cell_area,
            "U1_grid": U1_grid,
            "U2_grid": U2_grid,
            "U_grid": U_grid,
            "log_post": log_post,
            "log_post_grid": log_post_grid,
            "log_post_upper_bound": log_post_upper_bound,
            "axis_labels": par_names,
            "n_grid_1d": n_test_grid_1d,
            "n_grid": U_grid.shape[0]
        }

    def store_gp_pred(self, store_pred_prior=False, store_pred_clipped=False):
        """
        Storing predictions as Gaussian distributions. Wrapping in custom
        Gaussian class. Optionally store clipped Gaussian
        predictions as well.
        """
        U = self.test_grid_info["U_grid"]
        upper_bound = self.log_post_upper_bound(U)

        # Prior predictions
        if store_pred_prior:
            prior_latent = self.gp_posterior.prior.predict(U)
            prior_pred = self.gp_posterior.likelihood(prior_latent)
        else:
            prior_latent = None
            prior_pred = None
        
        # Conditional predictions
        post_latent = self.gp_posterior.predict(U, train_data=self.design)
        post_pred = self.gp_posterior.likelihood(post_latent)
        
        # Optionally store Clipped Gaussian predictions
        if store_pred_clipped:
            if store_pred_prior:
                prior_pred_clip = ClippedGaussian(prior_pred.mean, 
                                                    prior_pred.covariance_matrix,
                                                    upper=upper_bound,
                                                    rng=self.inv_prob.rng)
            else:
                prior_pred_clip = None
            post_pred_clip = ClippedGaussian(post_pred.mean, 
                                               post_pred.covariance_matrix,
                                               upper=upper_bound,
                                               rng=self.inv_prob.rng)
        else:
            prior_pred_clip = None
            post_pred_clip = None

        # Wrap as `Gaussian`
        if store_pred_prior:
            prior_latent = Gaussian(prior_latent.mean, prior_latent.covariance_matrix, rng=self.inv_prob.rng)
            prior_pred = Gaussian(prior_pred.mean, prior_pred.covariance_matrix, rng=self.inv_prob.rng)
        post_latent = Gaussian(post_latent.mean, post_latent.covariance_matrix, rng=self.inv_prob.rng)
        post_pred = Gaussian(post_pred.mean, post_pred.covariance_matrix, rng=self.inv_prob.rng)

        # self.gp_prior_pred = {"latent": prior_latent, "pred": prior_pred, "pred_clip": prior_pred_clip}
        self.gp_post_pred = {"latent": post_latent, "pred": post_pred, "pred_clip": post_pred_clip}

    def store_posterior_estimates(self, store_pred_clipped=True):
        """
        Store log normalized and unnormalized posterior estimates at grid points.
        """

        # Exact
        log_post_exact_norm, log_post_exact = self._exact_post_grid(shape_to_grid=False, return_log=True)
        
        # Plug-in mean
        log_post_mean_norm, log_post_mean = self._mean_approx_grid(shape_to_grid=False, return_log=True, pred_type='pred')
        if store_pred_clipped:
            log_post_mean_norm_clip, log_post_mean_clip = self._mean_approx_grid(shape_to_grid=False, 
                                                                                 return_log=True, 
                                                                                 pred_type='pred_clip')
        else:
            log_post_mean_norm_clip = log_post_mean_clip = None
        
        # eup
        log_post_eup_norm, log_post_eup = self._eup_approx_grid(shape_to_grid=False, return_log=True, pred_type='pred')
        if store_pred_clipped:
            log_post_eup_norm_clip, log_post_eup_clip = self._eup_approx_grid(shape_to_grid=False, 
                                                                               return_log=True, 
                                                                               pred_type='pred_clip')
        else:
            log_post_eup_norm_clip = log_post_eup_clip = None
        
        # ep
        log_post_ep_norm, log_post_ep = self._ep_approx_grid(shape_to_grid=False, return_log=True, pred_type='pred')
        if store_pred_clipped:
            log_post_ep_norm_clip, log_post_ep_clip = self._ep_approx_grid(shape_to_grid=False, return_log=True, pred_type='pred_clip')
        else:
            log_post_ep_norm_clip = log_post_ep_clip = None

        exact = {'log_norm': log_post_exact_norm, 
                 'log_unnorm': log_post_exact}
        mean = {'log_norm': log_post_mean_norm,
                'log_unnorm': log_post_mean,
                'log_norm_clip': log_post_mean_norm_clip,
                'log_unnorm_clip': log_post_mean_clip}
        eup = {'log_norm': log_post_eup_norm,
               'log_unnorm': log_post_eup,
               'log_norm_clip': log_post_eup_norm_clip,
               'log_unnorm_clip': log_post_eup_clip}
        ep = {'log_norm': log_post_ep_norm,
              'log_unnorm': log_post_ep,
              'log_norm_clip': log_post_ep_norm_clip,
              'log_unnorm_clip': log_post_ep_clip}
        
        self.posterior_estimates = {'exact': exact, 'mean': mean, 'eup': eup, 'ep': ep}


    def construct_design(self, design_method):
        prior = self.inv_prob.prior

        if design_method == 'lhc':
            x_design = prior.sample_lhc(self.n_design)
        elif design_method == 'uniform':
            x_design = prior.sample(self.n_design)
        else:
            raise ValueError(f'Invalid design method {design_method}')

        x_design = jnp.asarray(x_design)
        y_design = jnp.asarray(self.inv_prob.log_posterior_density(x_design)).reshape((-1,1))
        return gpx.Dataset(X=x_design, y=y_design)

    def construct_gp_mean(self):
        constant_param = gpx.parameters.Real(value=self.design.y.mean())
        meanf = gpx.mean_functions.Constant(constant_param)
        return meanf

    def construct_gp_kernel(self):
        lengthscales_init = average_pairwise_distance_per_dim(self.design.X)
        lengthscales_init = jnp.array(lengthscales_init)

        ker_var_init = gpx.parameters.PositiveReal(self.design.y.var())
        kernel = gpx.kernels.RBF(lengthscale=lengthscales_init, variance=ker_var_init, n_dims=self.inv_prob.dim)
        return kernel

    def construct_gp_prior(self):
        meanf = self.construct_gp_mean()
        kernel = self.construct_gp_kernel()
        gp_prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)
        return gp_prior
    
    def construct_gp_likelihood(self):
        obs_stddev = gpx.parameters.PositiveReal(self.gp_prior.jitter)
        gp_likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.design.n, obs_stddev=obs_stddev)
        return gp_likelihood

    def train_gp_hyperpars(self):
        starting_mll = -gpx.objectives.conjugate_mll(self.gp_untuned_posterior, self.design)

        gp_posterior, history = gpx.fit_scipy(
            model=self.gp_untuned_posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.design,
            trainable=gpx.parameters.Parameter,
        )

        ending_mll = -gpx.objectives.conjugate_mll(gp_posterior, self.design)
        opt_info = {"starting_mll": starting_mll,
                    "ending_mll": ending_mll,
                    "history": history}
        return gp_posterior, opt_info
    
    def predict(self, u, pred=None, clip=False):
        """ 
        Return Gaussian or Clipped Gaussian representing predictions 
        (including observation noise) at u 
        """
        if pred is None:
            latent = self.gp_posterior.predict(u, train_data=self.design)
            pred = self.gp_posterior.likelihood(latent)

            if clip:
                pred = ClippedGaussian(pred.mean, pred.covariance_matrix,
                                         upper=self.log_post_upper_bound(u),
                                         rng=self.inv_prob.rng)
            else:
                pred = Gaussian(pred.mean, pred.covariance_matrix, rng=self.inv_prob.rng)

        return pred

    def log_post_upper_bound(self, u: Array) -> Array:
        return self.inv_prob.likelihood.log_density_upper_bound(u) + self.inv_prob.prior.log_density(u)

    def log_post_approx_mean(self, u, pred=None, clip=False):
        """ Log of plug-in mean approximation of unnormalized posterior density. """
        pred = self.predict(u, pred, clip=clip)
        return pred.mean

    def log_post_approx_eup(self, u, pred=None, clip=False):
        """ Log of EUP approximation of unnormalized posterior density. """
        pred = self.predict(u, pred, clip=False)
         
        if clip:
            upper_bound = self.log_post_upper_bound(u)
            return log_mean_capped_lognormal(pred.mean, pred.variance, b=upper_bound)
        else:
            log_ln_mean = pred.mean + 0.5 * pred.variance ** 2
            return log_ln_mean

    def log_post_approx_ep(self, u, n_mc=10000, pred=None, clip=False):
        """ Log of EP approximation of normalized posterior density (approximate).
        The grid of test points is used to approximate the normalizing constants Z(f).
        The expectation with respect to f is estimated via simple Monte Carlo, by 
        sampling discretizations of f at the test points. `n_mc` is the number of 
        Monte Carlo samples to use.
        """
        pred = self.predict(u, pred, clip=clip)
        n_grid = pred.dim

        # log_post_samp[i,j] = log pi(u_j; f_i)
        log_post_samp = pred.sample(n_mc) # (n_mc, n_grid)

        cell_area = self.test_grid_info['cell_area']
        _, log_ep_approx = estimate_ep_grid(log_post_samp, cell_area)
        return log_ep_approx
    
    def _exact_post_grid(self, shape_to_grid=True, return_log=False):
        """ Exact posterior density evaluated at test grid 
        
        Returns tuple, with:
            - normalized posterior (log scale if `return_log` is True)
            - log unnormalized posterior (always on log scale)
        """
        U = self.test_grid_info['U_grid']
        log_unnormalized_post = self.test_grid_info['log_post']
        cell_area = self.test_grid_info['cell_area']
        normalized_post, _ = normalize_over_grid(log_unnormalized_post, cell_area, 
                                                 return_log=return_log)
        normalized_post = normalized_post.flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            normalized_post = normalized_post.reshape(n_grid_1d, n_grid_1d)
            log_unnormalized_post = log_unnormalized_post.reshape(n_grid_1d, n_grid_1d)
            return normalized_post, log_unnormalized_post
        else:
            return normalized_post, log_unnormalized_post

    def _mean_approx_grid(self, shape_to_grid=True, return_log=False, pred_type='pred'):
        """ Normalized plug-in mean approximation evaluated at test grid 
        
        `pred_type` is either 'latent', 'pred', or 'pred_clip'
        """
        U = self.test_grid_info['U_grid']
        cell_area = self.test_grid_info['cell_area']
        pred = self.gp_post_pred[pred_type]
        log_unnormalized_approx = self.log_post_approx_mean(U, pred=pred) # (n_grid,)
        normalized_approx, _ = normalize_over_grid(log_unnormalized_approx, cell_area, return_log=return_log)
        normalized_approx = normalized_approx.flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            normalized_approx = normalized_approx.reshape(n_grid_1d, n_grid_1d)
            log_unnormalized_approx = log_unnormalized_approx.reshape(n_grid_1d, n_grid_1d)
            return normalized_approx, log_unnormalized_approx
        else:
            return normalized_approx, log_unnormalized_approx

    def _eup_approx_grid(self, shape_to_grid=True, return_log=False, pred_type='pred'):
        """ Normalized EUP approximation evaluated at test grid """
        U = self.test_grid_info['U_grid']
        cell_area = self.test_grid_info['cell_area']

        # log_post_approx_eup requires `pred` to be Gaussian; handles transform internally
        pred = self.gp_post_pred['pred']
        clip = True if pred_type=='pred_clip' else False 

        log_unnormalized_approx = self.log_post_approx_eup(U, pred=pred, clip=clip) # (n_grid,)
        normalized_approx, _ = normalize_over_grid(log_unnormalized_approx, cell_area, return_log=return_log)
        normalized_approx = normalized_approx.flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            normalized_approx = normalized_approx.reshape(n_grid_1d, n_grid_1d)
            log_unnormalized_approx = log_unnormalized_approx.reshape(n_grid_1d, n_grid_1d)
            return normalized_approx, log_unnormalized_approx
        else:
            return normalized_approx, log_unnormalized_approx

    def _ep_approx_grid(self, shape_to_grid=True, return_log=False, pred_type='pred'):
        """ Normalized EP approximation evaluated at test grid """
        U = self.test_grid_info['U_grid']
        cell_area = self.test_grid_info['cell_area']
        pred = self.gp_post_pred[pred_type]
        log_unnormalized_approx = self.log_post_approx_ep(U, pred=pred) # (n_grid,)
        normalized_approx, _ = normalize_over_grid(log_unnormalized_approx, cell_area, return_log=return_log)
        normalized_approx = normalized_approx.flatten()

        if shape_to_grid:
            n_grid_1d = self.test_grid_info['n_grid_1d']
            normalized_approx = normalized_approx.reshape(n_grid_1d, n_grid_1d)
            log_unnormalized_approx = log_unnormalized_approx.reshape(n_grid_1d, n_grid_1d)
            return normalized_approx, log_unnormalized_approx
        else:
            return normalized_approx, log_unnormalized_approx

    def compute_metrics(self, pred_type='pred', probs=None):
        tag = 'log_norm_clip' if pred_type=='pred_clip' else 'log_norm'
        log_post_exact = self.posterior_estimates['exact']['log_norm']
        log_post_mean = self.posterior_estimates['mean'][tag]
        log_post_eup = self.posterior_estimates['eup'][tag]
        log_post_ep = self.posterior_estimates['ep'][tag]

        mean_kl = kl_grid(log_post_exact, log_post_mean)
        eup_kl = kl_grid(log_post_exact, log_post_eup)
        ep_kl = kl_grid(log_post_exact, log_post_ep)
        ep_eup_kl = kl_grid(log_post_ep, log_post_eup)
        ep_mean_kl = kl_grid(log_post_ep, log_post_mean)

        if probs is None:
            probs = np.linspace(0.1, 0.99, 99)

        def get_coverage(approx):
            res = coverage_curve_grid(
                log_post_exact,
                approx,
                probs=probs,
                cell_area=None, # since densities are already normalized
                return_masks=False,
                return_weights=False,
                tie_break='expand'
            )
            return np.exp(res[1])
        
        cover = np.empty((3, len(probs)))
        for i,approx in enumerate([log_post_mean, log_post_eup, log_post_ep]):
            cover[i] = get_coverage(approx)
        cover_labels = ['mean', 'eup', 'ep']

        kl_metrics = {
            'exact_mean': mean_kl,
            'exact_eup': eup_kl,
            'exact_ep': ep_kl,
            'ep_eup': ep_eup_kl,
            'ep_mean': ep_mean_kl
        }

        cover_metrics = {
            'labels': cover_labels,
            'probs': probs,
            'cover': cover
        }

        log_approx_metrics = self.compute_log_approx_metrics(pred_type=pred_type)

        return {
            'kl': kl_metrics,
            'coverage': cover_metrics,
            'log_approx': log_approx_metrics
        }

    def compute_log_approx_metrics(self, pred_type='pred'):
        """
        Average pointwise error between log normalized density approximation
        and true log normalized posterior.
        """

        tag = 'log_norm_clip' if pred_type=='pred_clip' else 'log_norm'
        log_post_exact = self.posterior_estimates['exact']['log_norm']
        log_post_mean = self.posterior_estimates['mean'][tag]
        log_post_eup = self.posterior_estimates['eup'][tag]
        log_post_ep = self.posterior_estimates['ep'][tag]

        return {
            'mae_prior_mean': _calc_mae(log_post_exact, log_post_mean),
            'mae_prior_eup': _calc_mae(log_post_exact, log_post_eup),
            'mae_prior_ep': _calc_mae(log_post_exact, log_post_ep),
            'mae_post_mean': _calc_mae(log_post_exact, log_post_mean, log_wt=log_post_exact),
            'mae_post_eup': _calc_mae(log_post_exact, log_post_eup, log_wt=log_post_exact),
            'mae_post_ep': _calc_mae(log_post_exact, log_post_ep, log_wt=log_post_exact)
        }

    def _get_plot_grid(self):
        grid_info = self.test_grid_info

        return grid_info["U1_grid"], grid_info["U2_grid"]

    def plot_exact_log_post(self):
        U1, U2 = self._get_plot_grid()
        log_post = self.test_grid_info["log_post_grid"]
        xlab, ylab = self.test_grid_info["axis_labels"]

        mappable, fig = plot_heatmap(U1, U2, log_post, title="Exact Posterior Log Density", 
                                     xlabel=xlab, ylabel=ylab)
        ax = fig.axes[0]
        ax.plot(*self.inv_prob.likelihood.par_true, "*", color="red", markersize=12)

        return fig, ax
    
    def plot_gp_bias(self, conditional=True, pred_type='pred', markersize=8, **kwargs):
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[pred_type].mean
        exact = self.test_grid_info["log_post"]
        biases = (means - exact).reshape(n,n)

        mappable, fig = plot_heatmap(U1, U2, biases, title="Emulator Bias", 
                                     xlabel=xlab, ylabel=ylab)
        ax = fig.axes[0]
        ax.plot(*self.inv_prob.likelihood.par_true, "*", color="red", markersize=12)
        ax.plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, ax
        

    def plot_gp_pred(self, conditional=True, pred_type='pred', markersize=8, **kwargs):
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[pred_type].mean.reshape(n,n)
        stdevs = jnp.sqrt(pred_dist[pred_type].variance).reshape(n,n)

        fig, axs, mappables = plot_independent_heatmaps(
            U1, U2,
            Z_list=[means, stdevs],
            titles=[f"{pred_type} mean", f"{pred_type} std dev"],
            xlab=xlab, ylab=ylab,
            **kwargs
        )

        # Add design points
        axs[0].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)
        axs[1].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs


    def plot_true_vs_gp_mean(self, conditional=True, latent_pred=False, markersize=8, **kwargs):
        dist_label = "latent" if latent_pred else "pred"
        pred_dist = self.gp_post_pred if conditional else self.gp_prior_pred

        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]
        means = pred_dist[dist_label].mean.reshape(n,n)
        log_post_true = self.test_grid_info["log_post_grid"]

        fig, axs, mappables, cbar_obj = plot_shared_scale_heatmaps(
            U1, U2,
            Z_list=[log_post_true, means],
            titles=["exact log posterior", f"{dist_label} GP mean"],
            xlab=xlab, ylab=ylab,
            sharexy=True,
            **kwargs
        )

        # Add design points
        axs[0].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)
        axs[1].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs


    def plot_posterior_comparison(self, markersize=8, shared_scale=True, log_scale=False, 
                                  pred_type='pred', **kwargs):
        """ Plot exact vs plug-in mean vs EUP vs EP normalized densities """
        U1, U2 = self._get_plot_grid()
        n = U1.shape[0]
        xlab, ylab = self.test_grid_info["axis_labels"]

        exact, _ = self._exact_post_grid(return_log=log_scale)
        mean, _ = self._mean_approx_grid(return_log=log_scale, pred_type=pred_type)
        eup, _ = self._eup_approx_grid(return_log=log_scale, pred_type=pred_type)
        ep, _ = self._ep_approx_grid(return_log=log_scale, pred_type=pred_type)

        param_list = [U1, U2]
        param_dict = {
            'Z_list': [exact, mean, eup, ep],
            'titles': ["exact", "mean", "eup", "ep"],
            'xlab': xlab,
            'ylab': ylab
        }

        if shared_scale:
            param_dict['sharexy'] = True
            fig, axs, mappables, cbar_obj = plot_shared_scale_heatmaps(*param_list, **param_dict, **kwargs)
        else:
            fig, axs, mappables = plot_independent_heatmaps(*param_list, **param_dict, **kwargs)

        # Add design points
        for i in range(4):
            axs[i].plot(self.design.X[:,0], self.design.X[:,1], "o", color="red", markersize=markersize)

        return fig, axs
    

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _calc_mae(x, y, log_wt=None):
    """
    Compute mean absolute error between x and y, optionally weighted
    by `np.exp(log_wt)`. Note that the log of the weights must be passed.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError('_calc_mae() requires x,y to have equal length.')

    if log_wt is None:
        # default to equal weights 1/n
        log_wt = np.full(n, -np.log(n))

    abs_diff = np.abs(x - y)
    log_summands = np.log(abs_diff) + log_wt
    log_mae = logsumexp(log_summands).item()

    return np.exp(log_mae)


def average_pairwise_distance(x):
    """ x: (n, d) array. 
    Computes average Euclidean distance over set of pairwise distinct points
    """
    n = x.shape[0]
    diffs = x[:, None, :] - x[None, :, :]  # Shape: (n, n, d)
    dists = np.linalg.norm(diffs, axis=-1) # Shape: (n, n)

    # Distances between *distinct* pairs, so mask the diagonal
    mask = ~np.eye(n, dtype=bool)
    avg_dist = dists[mask].mean()
    return avg_dist


def average_pairwise_distance_per_dim(x):
    n, d = x.shape
    diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
    abs_diffs = np.abs(diffs)

    # Discard diagonal (i==j), only keep i != j pairs
    mask = ~np.eye(n, dtype=bool)
    abs_diffs_pairs = abs_diffs[mask].reshape(n * (n - 1), d)
    avg_dist_per_dim = abs_diffs_pairs.mean(axis=0)
    return avg_dist_per_dim


def plot_heatmap(X, Y, Z, title=None, ax=None,
                 cmap='viridis', shading='auto',
                 xlabel=None, ylabel=None,
                 cbar=True, cbar_kwargs=None):
    """
    Plot a single heatmap and return the mappable (QuadMesh/AxesImage).
    If ax is None, create a new figure+axis.

    Parameters
    ----------
    X, Y : 2D arrays
        Grid coordinates (as used by pcolormesh).
    Z : 2D array
        Values to plot; must match X/Y shape.
    title : str or None
    ax : matplotlib.axes.Axes or None
    cmap : str
    shading : str
    xlabel, ylabel : str or None
    cbar : bool
    cbar_kwargs : dict passed to fig.colorbar (optional)

    Returns
    -------
    mappable : the object returned by pcolormesh
    fig_or_none : If a new figure was created, returns that figure, otherwise None
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots()

    # pcolormesh expects arrays shaped like the mesh
    m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if cbar:
        cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
        orientation = cbar_kwargs.pop('orientation', 'horizontal')
        fraction = cbar_kwargs.pop('fraction', 0.07)
        pad = cbar_kwargs.pop('pad', 0.15)
        fig_for_cb = created_fig if created_fig is not None else ax.figure
        cb = fig_for_cb.colorbar(m, ax=ax, orientation=orientation,
                                 fraction=fraction, pad=pad, **cbar_kwargs)
    else:
        cb = None

    return m, created_fig  # created_fig is None if ax was passed in


def plot_independent_heatmaps(X, Y, Z_list, titles=None,
                              xlab=None, ylab=None,
                              nrows=1, ncols=None, figsize=None,
                              cmap='viridis', shading='auto',
                              sharexy=False, **kwargs):
    """
    Plot multiple independent heatmaps (each with its own colorbar).
    Returns (fig, axs, mappables) and does NOT call plt.show().

    Parameters
    ----------
    X, Y : 2D arrays (same grid for all Zs)
    Z_list : list/iterable of 2D arrays (each same shape as X/Y)
    titles : list of str or None
    xlab, ylab : labels (applied to all subplots if provided)
    nrows : int
    ncols : int or None (auto computed if None)
    figsize : tuple or None (auto computed if None)
    cmap, shading : passed to pcolormesh
    sharexy : if True, share x/y axes between subplots (useful for aligned grids)

    Returns
    -------
    fig, axs_flat, mappables_list
    """
    nplots = len(Z_list)
    if ncols is None:
        ncols = int(ceil(nplots / nrows))

    # sensible default figure size if not provided:
    # give each subplot ~4 x 3 inches
    if figsize is None:
        per_ax_w, per_ax_h = 4.0, 3.0
        figsize = (per_ax_w * ncols, per_ax_h * nrows)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                            sharex=sharexy, sharey=sharexy)
    axs_flat = axs.reshape(-1)

    # normalize titles list length
    if titles is None:
        titles = [None] * nplots
    else:
        # extend or truncate to match nplots
        titles = list(titles) + [None] * max(0, nplots - len(titles))
        titles = titles[:nplots]

    mappables = []
    for i, Z in enumerate(Z_list):
        ax = axs_flat[i]
        m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap)
        mappables.append(m)

        if titles[i] is not None:
            ax.set_title(titles[i])
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)

        # add a horizontal colorbar under this axis with reasonable defaults
        cb = fig.colorbar(m, ax=ax, orientation='horizontal',
                          fraction=0.07, pad=0.18)

    # Hide any unused axes
    total_axes = nrows * ncols
    for j in range(nplots, total_axes):
        fig.delaxes(axs_flat[j])

    # Use tight_layout but leave space for colorbars; caller may further adjust
    fig.tight_layout()
    return fig, axs_flat[:nplots], mappables


def plot_shared_scale_heatmaps(
    X: np.ndarray,
    Y: np.ndarray,
    Z_list: Iterable[np.ndarray],
    titles: Optional[Iterable[Optional[str]]] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    nrows: int = 1,
    ncols: Optional[int] = None,
    figsize: Optional[tuple[float, float]] = None,
    cmap: str = 'viridis',
    shading: str = 'auto',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar: bool = True,
    cbar_kwargs: Optional[dict] = None,
    sharexy: bool = False,
    **kwargs
):
    """
    Plot multiple heatmaps that share a single global color scale (one colorbar).

    Parameters
    ----------
    X, Y : 2D arrays
        Grid coordinates (mesh) matching Z shapes (as used by pcolormesh).
    Z_list : iterable of 2D arrays
        Data arrays to plot; each must have the same shape as X/Y.
    titles : list/iterable of str or None
        Titles for each subplot (length will be truncated/extended to match the number of Zs).
    xlab, ylab : str or None
        Common x and y labels applied to all subplots if provided.
    nrows : int
        Number of subplot rows.
    ncols : int or None
        Number of subplot columns; auto-computed if None.
    figsize : (w, h) or None
        Figure size in inches; auto-computed if None (approx 4x3 inches per subplot).
    cmap : str
        Matplotlib colormap name.
    shading : str
        Passed to pcolormesh.
    vmin, vmax : float or None
        Global color limits. If None, computed from min/max of all Zs.
    cbar : bool
        Whether to draw the shared colorbar.
    cbar_kwargs : dict or None
        Extra kwargs passed to `fig.colorbar`. Common keys: orientation, fraction, pad, label, shrink, extend.
    sharexy : bool
        If True, share x and y axes between subplots.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs_flat : array-like of Axes (length == nplots)
        Flat array of axes corresponding to each plot (in row-major order).
    mappables : list
        list of QuadMesh objects returned by pcolormesh for each subplot.
    cbar_obj : matplotlib.colorbar.Colorbar or None
        The shared colorbar object (None if cbar=False).
    """
    Z_list = list(Z_list)
    nplots = len(Z_list)
    if nplots == 0:
        raise ValueError("Z_list must contain at least one array to plot.")

    # compute ncols if necessary
    if ncols is None:
        ncols = int(ceil(nplots / nrows))

    # reasonable default figure size if not provided
    if figsize is None:
        per_ax_w, per_ax_h = 4.0, 3.0
        figsize = (per_ax_w * ncols, per_ax_h * nrows)

    # compute global vmin/vmax if not provided
    if vmin is None:
        vmin = min(np.nanmin(Z) for Z in Z_list)
    if vmax is None:
        vmax = max(np.nanmax(Z) for Z in Z_list)

    # prepare titles
    if titles is None:
        titles = [None] * nplots
    else:
        titles = list(titles) + [None] * max(0, nplots - len(list(titles)))
        titles = titles[:nplots]

    # create subplots
    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, squeeze=False,
        sharex=sharexy, sharey=sharexy
    )
    axs_flat = axs.reshape(-1)

    mappables = []
    for i, Z in enumerate(Z_list):
        ax = axs_flat[i]
        m = ax.pcolormesh(X, Y, Z, shading=shading, cmap=cmap, vmin=vmin, vmax=vmax)
        mappables.append(m)

        if titles[i] is not None:
            ax.set_title(titles[i])
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)

    # hide unused axes
    total_axes = nrows * ncols
    for j in range(nplots, total_axes):
        fig.delaxes(axs_flat[j])

    # add a single shared colorbar (span all axes)
    cbar_obj = None
    if cbar:
        cbar_kwargs = {} if cbar_kwargs is None else dict(cbar_kwargs)
        # defaults for a horizontal colorbar placed beneath the axes
        orientation = cbar_kwargs.pop('orientation', 'horizontal')
        fraction = cbar_kwargs.pop('fraction', 0.07)
        pad = cbar_kwargs.pop('pad', 0.18)
        # use the first mappable as the reference, and provide all axes for placement
        axes_for_cb = axs_flat[:nplots]  # only existing axes
        cbar_obj = fig.colorbar(
            mappables[0],
            ax=axes_for_cb,
            orientation=orientation,
            fraction=fraction,
            pad=pad,
            **cbar_kwargs
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18) # extra space for colorbar     
    # leave caller freedom to further adjust (e.g. fig.subplots_adjust(bottom=...))
    return fig, axs_flat[:nplots], mappables, cbar_obj


def estimate_ep_grid(logpi_samples: Array, cell_area: float) -> tuple[Array, Array]:
    """
    Estimate E_f[ pi(u_j; f) / Z(f) ] at grid nodes using Monte Carlo samples of log-densities.

    Args:
        logpi_samples: ndarray of shape (S, M). Row s contains [ell_j(f^{(s)})]_{j=1..M},
                       where ell_j = log pi(u_j; f^{(s)}).
                       S = number of Monte Carlo samples, M = number of grid nodes.
        cell_area: area of one grid cell, assumed constant across cells.

    Returns:
        ep: ndarray of shape (M,), the Monte Carlo estimate of E_f[ pi(u_j;f)/Z(f) ].
        log_mean_p: log of ep
    """
    logp, _ = normalize_over_grid(logpi_samples, cell_area, return_log=True)
    n_densities, _ = logp.shape

    # log of normalized ep density 
    log_ep = logsumexp(logp.T) - np.log(n_densities) # (n_densities,)

    return np.exp(log_ep), log_ep


def log_mean_capped_lognormal(m, s2, b):
    """
    Compute log E[min(exp(X), exp(b))] for X ~ N(m, s^2).

    Parameters
    ----------
    m : array_like
        Means of the normal distribution.
    s2 : array_like
        Variances of the normal distribution.
    b : array_like
        Log-cap values: y = min(exp(x), exp(b)).

    Returns
    -------
    log_Ey : ndarray
        log( E[min(exp(X), exp(b))] ), elementwise.
    """
    m = np.asarray(m)
    s2 = np.asarray(s2)
    b = np.asarray(b)

    s = np.sqrt(s2)

    # z1 = (b - (m + s^2)) / s
    z1 = (b - (m + s2)) / s

    # z2 = (b - m) / s
    z2 = (b - m) / s

    # Log of the two contributions:
    # T1 = exp(m + s^2/2) * Phi(z1)
    logT1 = (m + 0.5 * s2) + norm.logcdf(z1)

    # T2 = exp(b) * (1 - Phi(z2)) = exp(b) * SF(z2)
    logT2 = b + norm.logsf(z2)

    # log(E[y]) = logsumexp(logT1, logT2)
    # logsumexp for two terms:
    M = np.maximum(logT1, logT2)
    log_Ey = M + np.log(np.exp(logT1 - M) + np.exp(logT2 - M))

    return log_Ey