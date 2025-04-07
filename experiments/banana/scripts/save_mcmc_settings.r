# 
# save_mcmc_settings.r
# Defines the approximate MCMC algorithms that will be run and saves the 
# settings to file. The saved CSV file is read in by `run_approx_mcmc.r`. 
#
# Andrew Roberts
#

#
# TODO: Create an "adapt_settings" sublist, similar to "ic_settings".
#

library(data.table)

# Output directory and path for saving settings file.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
out_dir <- file.path(base_dir, "output", "gp_inv_prob", "vsem")
out_path <- file.path(out_dir, "mcmc_approx_settings.rds")

# Common settings that will be applied to all MCMC algorithms.
common_settings <- list(n_itr=150000L, try_parallel=FALSE, n_chain=4L,
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
# `mcmc_gp_acc_prob_approx()`.
# common_acc_prob_settings <- list(mcmc_func_name="mcmc_gp_acc_prob_approx", 
#                                  mode="marginal",
#                                  ic_settings=list(approx_type="quantile", 
#                                                   n_test_inputs=500L,
#                                                   alpha=0.8,
#                                                   n_ic_by_method=c(design_max=2, 
#                                                                    approx_max=2)))

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
  c(list(test_label="mcwmh-joint-trunc", mode="mcwmh", use_joint=TRUE, adjustment="truncated"),
    common_settings, common_noisy_settings),
  c(list(test_label="mcwmh-ind-rect", mode="mcwmh", use_joint=FALSE, adjustment="rectified"),
    common_settings, common_noisy_settings),
  c(list(test_label="mcwmh-ind-trunc", mode="mcwmh", use_joint=FALSE, adjustment="truncated"),
    common_settings, common_noisy_settings),
  c(list(test_label="pm-joint-rect", mode="pseudo-marginal", use_joint=TRUE, adjustment="rectified"),
    common_settings, common_noisy_settings),
  c(list(test_label="pm-ind-rect", mode="pseudo-marginal", use_joint=FALSE, adjustment="rectified"),
    common_settings, common_noisy_settings)
)


# mcmc_settings_list <- list(
#   c(list(test_label="quantile7", approx_type="quantile", alpha=0.7),
#     common_settings, common_dens_settings),
#   c(list(test_label="quantile8", approx_type="quantile", alpha=0.8),
#     common_settings, common_dens_settings),
#   c(list(test_label="quantile9", approx_type="quantile", alpha=0.9),
#     common_settings, common_dens_settings),
#   c(list(test_label="mean", approx_type="mean"),
#     common_settings, common_dens_settings),
#   c(list(test_label="marginal", approx_type="marginal"),
#     common_settings, common_dens_settings),
#   c(list(test_label="mcwmh-joint", mode="mcwmh", use_joint=TRUE),
#     common_settings, common_noisy_settings),
#   c(list(test_label="mcwmh-ind", mode="mcwmh", use_joint=FALSE),
#     common_settings, common_noisy_settings),
#   c(list(test_label="pm-joint", mode="pseudo-marginal", use_joint=TRUE),
#     common_settings, common_noisy_settings),
#   c(list(test_label="pm-ind", mode="pseudo-marginal", use_joint=FALSE),
#     common_settings, common_noisy_settings),
#   c(list(test_label="marg-acc-prob-joint", use_joint=TRUE), 
#     common_settings, common_acc_prob_settings),
#   c(list(test_label="marg-acc-prob-ind", use_joint=FALSE), 
#     common_settings, common_acc_prob_settings)
# )

# Set names of list to the MCMC tags.
tags <- sapply(mcmc_settings_list, function(x) x$test_label)
names(mcmc_settings_list) <- tags

# Fine-tune initial condition generation for specific algorithms.
mcmc_settings_list[["pm-ind-rect"]]$ic_settings$approx_type <- "marginal"
mcmc_settings_list[["pm-ind-rect"]]$ic_settings$adjustment <- "rectified"

# mcmc_settings_list[["quantile7"]]$ic_settings[c("approx_type","alpha")] <- list("quantile", 0.7)
# mcmc_settings_list[["quantile8"]]$ic_settings[c("approx_type","alpha")] <- list("quantile", 0.8)
# mcmc_settings_list[["quantile9"]]$ic_settings[c("approx_type","alpha")] <- list("quantile", 0.9)
mcmc_settings_list[["mean-rect"]]$ic_settings$approx_type <- "mean"
mcmc_settings_list[["mean-rect"]]$ic_settings$adjustment <- "rectified"

mcmc_settings_list[["marginal-rect"]]$ic_settings$approx_type <- "marginal"
mcmc_settings_list[["marginal-rect"]]$ic_settings$adjustment <- "rectified"

# Save settings to file.
saveRDS(mcmc_settings_list, out_path)





