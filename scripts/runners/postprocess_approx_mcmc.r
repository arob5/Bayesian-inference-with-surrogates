#
# postprocess_approx_mcmc.r
# This script is intended to be run after `run_approx_mcmc.r` to identify 
# potential issues with MCMC runs. See `run_approx_mcmc.r` for details on 
# specifying filepaths, etc., as this script uses the same setup. This script 
# should be run before `analyze_approx_mcmc.r`, as this latter script assumes 
# individual chains are well-mixed, burn-in has already been dropped, issues 
# corrected, etc.
# 
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Settings for defining what constitutes a valid MCMC run. The R-hat threshold
# is for the maximum R-hat over all parameters on a per-chain basis.
# The min itr threshold is the minimum number of samples a chain is allowed
# to have.
rhat_threshold <- 1.05
min_itr_threshold <- 500L

# Base directory: all paths are relative to this directory.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Directory to source code.
src_dir <- file.path(base_dir, "src")

# Set variables controlling filepaths.
experiment_tag <- "vsem"
round <- 1L

print(paste0("experiment_tag: ", experiment_tag))
print(paste0("round: ", round))

# Define directories
round_tag <- paste0("round", round)
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
mcmc_dir <- file.path(experiment_dir, round_tag, "mcmc")

# Ensure required paths exist.
mcmc_settings_path <- file.path(experiment_dir, "mcmc_approx_settings.rds")
inv_prob_path <- file.path(experiment_dir, "inv_prob_setup", "inv_prob_list.rds")

print("-----> Checking required files exist:")
assert_that(file.exists(mcmc_settings_path))
assert_that(file.exists(inv_prob_path))

print("-----> Settings:")
print(paste0("rhat_threshold: ", rhat_threshold))
print(paste0("min_itr_threshold: ", min_itr_threshold))

mcmc_settings_list <- readRDS(mcmc_settings_path)

# Source required scripts.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "inv_prob_test_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "gp_helper_functions.r"))
source(file.path(src_dir, "gpWrapper.r"))
source(file.path(src_dir, "llikEmulator.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(src_dir, "gp_mcmc_functions.r"))
source(file.path(src_dir, "sim_study_functions.r"))

# Load R project. 
# renv::load(base_dir)
# print(".libPaths()")
# print(.libPaths())
# renv::status()

# ------------------------------------------------------------------------------
# Define helper function to process a single MCMC run.
# ------------------------------------------------------------------------------

# TODO: 
#   write function that iteratively increases the burn-in (on a chain-by-chain)
#   basis until it is considered "valid". Should stop at a certain point 
#   so that a lower bound on the number of iterations is enforced (say, we 
#   shouldn't use a chain with fewer than 500 iterations).

mcmc_tags <- setdiff(list.files(mcmc_dir), "summary_files")
mcmc_summary <- data.table(mcmc_id = character(),
                           mcmc_tag = character(),
                           n_chains = integer(), 
                           max_rhat = integer(),
                           status = logical())
chain_summary <- data.table(mcmc_id = character(),
                            mcmc_tag = character(),
                            test_label = character(),
                            chain_idx = integer(),
                            rhat = numeric(),
                            itr_min = integer(),
                            itr_max = integer(),
                            llik_mean = numeric(),
                            llik_var = numeric(),
                            n_itr = integer(),
                            log_weight = numeric())
param_summary <- data.table(mcmc_id = character(),
                            mcmc_tag = character(),
                            test_label = character(),
                            chain_idx = integer(),
                            param_type = character(),
                            param_name = character(),
                            R_hat = numeric())


for(tag in mcmc_tags) {
  print(paste0("MCMC tag: ", tag))
  tag_dir <- file.path(mcmc_dir, tag)
  id_map <- fread(file.path(tag_dir, "id_map.csv"))
  mcmc_ids <- id_map$mcmc_id
  
  for(mcmc_id in mcmc_ids) {
    cat("\t", mcmc_id, "\n")
    samp_list <- try(readRDS(file.path(tag_dir, mcmc_id, "samp.rds")))
    
    if(class(samp_list) != "try-error") {
      summary_info <- process_mcmc_run(samp_list, rhat_threshold=rhat_threshold, 
                                       min_itr_threshold=min_itr_threshold)
      
      # High-level MCMC run information.
      summary <- summary_info$summary
      summary[, `:=`(mcmc_id=mcmc_id, mcmc_tag=tag)]
      mcmc_summary <- rbindlist(list(mcmc_summary, summary), use.names=TRUE)
      
      # Chain-by-chain information.
      chain_info <- summary_info$chain_info
      if(!is.null(chain_info)) {
        chain_info[, `:=`(mcmc_id=mcmc_id, mcmc_tag=tag)]
        chain_summary <- rbindlist(list(chain_summary, chain_info), use.names=TRUE)
      }
      
      # Rhat by parameter.
      rhat_info <- summary_info$rhat
      if(!is.null(rhat_info)) {
        rhat_info[, `:=`(mcmc_id=mcmc_id, mcmc_tag=tag)]
        param_summary <- rbindlist(list(param_summary, rhat_info), use.names=TRUE)
      }
      
    } else {
      summary <- data.table(mcmc_id=mcmc_id, mcmc_tag=tag, n_chains=0L,
                            max_rhat=NA, status="rds_read_error")
      mcmc_summary <- rbindlist(list(mcmc_summary, summary), use.names=TRUE)
    }
    
  }
}

mcmc_summary_dir <- file.path(mcmc_dir, "summary_files")
fwrite(mcmc_summary, file.path(mcmc_summary_dir, "mcmc_summary.csv"))
fwrite(chain_summary, file.path(mcmc_summary_dir, "chain_summary.csv"))
fwrite(param_summary, file.path(mcmc_summary_dir, "param_summary.csv"))








