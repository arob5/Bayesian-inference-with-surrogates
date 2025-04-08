# 
# save_alg_settings.r
# Defines the emulator models, MCMC algorithms, and sequential design methods
# that will be used throughout the experiment.
#
# Andrew Roberts
#

#
# TODO: Create an "adapt_settings" sublist, similar to "ic_settings".
#


get_emulator_settings <- function() {
  # Different emulator models are stored in a list. The names of the list
  # are used as the unique ID for each model. Each element of the list is
  # itself a list with elements "em_label", "is_fwd_em", and "fit_em". 
  # "label" serves as a unique identifier for the model. "is_fwd_em" is a
  # logical value indicating whether the model emulates the forward model 
  # or the log-likelihood. "fit_em" is a function that takes the arguments 
  # `design_info` and `inv_prob`, and returns an emulator object that has been 
  # fit to that design.
  
  em_list <- list()
  
  # Log-likelihood emulator. Gaussian kernel, constant mean.
  em_list$em_llik$em_label <- "em_llik"
  em_list$em_llik$is_fwd_em <- FALSE
  em_list$em_llik$fit_em <- function(design_info, inv_prob) {
    
    # Fit GP for log-likelihood.
    gp_obj <- gpWrapperHet(design_info$input, matrix(design_info$llik, ncol=1),
                           scale_input=TRUE, normalize_output=TRUE)
    gp_obj$set_gp_prior("Gaussian", "constant", include_noise=FALSE)
    gp_obj$fit()
    
    # Instantiate and save log-likelihood emulator object.
    llik_em <- llikEmulatorGP("em_llik", gp_obj, default_conditional=FALSE, 
                              default_normalize=TRUE, lik_par=inv_prob$sig2_model, 
                              llik_bounds=inv_prob$llik_obj$get_llik_bounds(),
                              input_names=inv_prob$par_names)
    
    return(llik_em)
  }
  
  # Forward model emulator. Gaussian kernel, constant mean.
  em_list$em_fwd$em_label <- "em_fwd"
  em_list$em_fwd$is_fwd_em <- TRUE
  em_list$em_fwd$fit_em <- function(design_info, inv_prob) {
    
    # Fit GP for forward model.
    gp_obj <- gpWrapperHet(design_info$input, design_info$fwd,
                           scale_input=TRUE, normalize_output=TRUE)
    gp_obj$set_gp_prior("Gaussian", "constant", include_noise=FALSE)
    gp_obj$fit()
    
    # Instantiate and save log-likelihood emulator object.
    llik_em <- llikEmulatorGPFwdGaussDiag("em_fwd", gp_obj,
                                          y_obs=inv_prob$llik_obj$y,
                                          sig2=inv_prob$sig2_model,
                                          default_conditional=FALSE,
                                          default_normalize=TRUE,
                                          par_names=inv_prob$par_names)

    return(llik_em)
  }
  
  return(em_list)
}


get_mcmc_settings <- function(experiment_dir) {
  # Returns list, each element of which defines an MCMC algorithm. Each of these
  # elements is itself a list containing the MCMC settings defining that 
  # algorithm. The "test_label" element of each of these lists is used as 
  # the unique identifier for the algorithm.
  
  # Common settings that will be applied to all MCMC algorithms.
  common_settings <- list(n_itr=150000L, try_parallel=TRUE, n_chain=4L,
                          itr_start=100000L)
  
  # Common settings that will be applied to all algorithms using the BayesianTools
  # wrapper function.
  common_bt_settings <- list(mcmc_func_name="mcmc_bt_wrapper", sampler="DEzs", 
                             defer_ic=TRUE, settings_list=list(consoleUpdates=25000))
  
  # Common settings that will be applied to all algorithms using the 
  # `mcmc_noisy_llik()`.
  common_noisy_settings <- list(mcmc_func_name="mcmc_noisy_llik", 
                                ic_settings=list(approx_type="quantile", 
                                                 n_test_inputs=500L,
                                                 alpha=0.8,
                                                 n_ic_by_method=c(design_max=2, 
                                                                  approx_max=2)))
  
  # Common settings that will be applied to all algorithms using the 
  # `mcmc_gp_unn_post_dens_approx()`.
  common_dens_settings <- list(mcmc_func_name="mcmc_gp_unn_post_dens_approx", 
                               ic_settings=list(approx_type="",
                                                n_test_inputs=500L,
                                                alpha=0.8,
                                                n_ic_by_method=c(design_max=2, 
                                                                 approx_max=2)))
  
  # List of MCMC settings.
  mcmc_settings_list <- list(
    c(list(test_label="mean-rect", approx_type="mean", adjustment="rectified"),
      common_settings, common_dens_settings),
    c(list(test_label="marginal-rect", approx_type="marginal", adjustment="rectified"),
      common_settings, common_dens_settings),
    c(list(test_label="mcwmh-joint-rect", mode="mcwmh", use_joint=TRUE, adjustment="rectified"),
      common_settings, common_noisy_settings),
    c(list(test_label="mcwmh-ind-rect", mode="mcwmh", use_joint=FALSE, adjustment="rectified"),
      common_settings, common_noisy_settings),
    c(list(test_label="pm-joint-rect", mode="pseudo-marginal", use_joint=TRUE, adjustment="rectified"),
      common_settings, common_noisy_settings),
    c(list(test_label="pm-ind-rect", mode="pseudo-marginal", use_joint=FALSE, adjustment="rectified"),
      common_settings, common_noisy_settings)
  )
  
  # Repeat all algorithms without the rectified adjustment.
  mcmc_settings_list_unadjusted <- mcmc_settings_list
  for(i in seq_along(mcmc_settings_list_unadjusted)) {
    mcmc_settings_list_unadjusted[[i]]$adjustment <- "none"
  }
  
  # Combine into single list.
  mcmc_settings_list <- c(mcmc_settings_list, mcmc_settings_list_unadjusted)
  
  return(mcmc_settings_list) 
}


save_design_settings <- function(experiment_dir) {
  # Returns a list, with each element a sublist defining a sequential design
  # method/acquisition function. The "acq_label" element of each sub-list
  # serves as a unique identifier for the method.
  
  # In this simple 2d example, we optimize over equally spaced grid-points.
  # Note that the "tensor_product_grid" method is deterministic, so this will
  # result in the same set of candidate points for all methods.
  candidate_settings <- list(n_prior=35^2, method="tensor_product_grid")

  # List that will be passed to acquisition functions. We pre-define two 
  # common cases here: one where no adjustment is applied to Gaussian predictive
  # distribution, and one where rectification adjustment is performed.
  acq_args_no_adj <- list(adjustment="none")
  acq_args_rect <- list(adjustment="rectified")

  # For acquisitions that require integrating over the design space, we define
  # different configurations of grid points used for numerical integration.
  # The number of integration grid points is partitioned between points sampled
  # from the prior and points sampled from an MCMC run, `n_prior` and `n_mcmc`,
  # respectively.
  prior_grid <- list(n_prior=200L, n_mcmc=0L, method="subsample", subsample_method="support")
  post_grid <- list(n_prior=0L, n_mcmc=200L, method="subsample", subsample_method="support")
  mix_grid <- list(n_prior=50L, n_mcmc=150L, method="subsample", subsample_method="support")
  
  # Baseline strategies that simply require sampling from prior or approximate
  # posterior (no optimization performed). Note that the `sample_ratios` will 
  # be normalized to sum to one - they do not determine the number of points
  # sampled, only the proportions.
  l_baseline <- list(
    list(acq_label="sample_prior", name="simple"), # Simple random sample from prior.
    list(acq_label="subsample_post", name="subsample", sample_ratios=post_grid),
    list(acq_label="subsample_mix", name="subsample", sample_ratios=mix_grid)
  )
  
  # Pure sequential design strategies (requires running forward model after
  # each individual acquisition).
  l_pure <- list(
    list(acq_label="ivar_prior", name="llik_IVAR_grid_gp", int_grid_settings=prior_grid),
    list(acq_label="ivar_post", name="llik_IVAR_grid_gp", int_grid_settings=post_grid),
    list(acq_label="ivar_mix", name="llik_IVAR_grid_gp", int_grid_settings=mix_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=prior_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=post_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=mix_grid),
    list(acq_label="neg_var_gp", name="llik_neg_var_gp"),
    list(acq_label="neg_var_lik", name="llik_neg_var_lik")
  )
  
  for(i in seq_along(l_pure)) {
    l_pure[[i]]$response_heuristic <- NA
    l_pure[[i]]$acq_args <- acq_args_no_adj
  }

  # Copy all of the pure acquisitions with rectified adjustment.
  l_pure_rect <- l_pure
  for(i in seq_along(l_pure_rect)) {
    l_pure_rect[[i]]$acq_args <- acq_args_rect
    old_label <- l_pure_rect[[i]]$acq_label
    l_pure_rect[[i]]$acq_label <- paste0(old_label, "_rect")
  }
  
  # Batch strategies. These will all use the rectified adjustment.
  l_batch <- l_pure
  for(i in seq_along(l_batch)) {
    l_batch[[i]]$response_heuristic <- "cl_mix"
    l_batch[[i]]$acq_args <- acq_args_rect
    old_label <- l_pure_rect[[i]]$acq_label
    l_batch[[i]]$acq_label <- paste0(old_label, "_clmix")
  }

  # Combine into single list and return.  
  l <- c(l_baseline, l_pure, l_pure_rect, l_batch)
  return(l)
}


