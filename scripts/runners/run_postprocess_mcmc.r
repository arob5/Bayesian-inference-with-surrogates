#
# postprocess_approx_mcmc.r
# This script is intended to be run after `run_approx_mcmc.r` to identify 
# potential issues with MCMC runs, as well as conduct MCMC post-processing.
# This includes computation of diagnostics (R-hat), determiming burn-ins,
# and computing chain weights for nonmixing chains. This script should be
# run before `run_analyze_mcmc.r`.
#
# The main inputs to this script are an experiment tag and round. It then
# processes all MCMC output found in:
# experiments/<experiment_tag>/round_<round>/mcmc
# 
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Paths
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Experiment and round.
experiment_tag <- "banana"
round <- 1L

# Settings for defining what constitutes a valid MCMC run. The R-hat threshold
# is for the maximum R-hat over all parameters with an MCMC run (or within
# a single MCMC chain for within-chain diagnostics used for nonmixing chains).
# The min itr threshold is the minimum number of samples a chain is allowed
# to have.
rhat_threshold <- 1.05
min_itr_threshold <- 500L


# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

round_name <- paste0("round", round)

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
mcmc_dir <- file.path(experiment_dir, "output", round_name, "mcmc")

# Paths.
mcmc_ids_path <- file.path(mcmc_dir, "id_map.csv")

# Source required scripts.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(base_dir, "scripts", "helper", "sim_study_functions.r"))

# ------------------------------------------------------------------------------
# Identify MCMC runs to process.
# ------------------------------------------------------------------------------

# Load MCMC ID map. Unique by columns "mcmc_id", "mcmc_tag".
# TODO: I messed up and rows are actually unique by "mcmc_id", "mcmc_tag",
# "em_tag". Need to fix this.
mcmc_ids <- fread(mcmc_ids_path)[, .(mcmc_tag, em_tag, em_id, mcmc_id)]
print(paste0("Preparing to process ", nrow(mcmc_ids), " MCMC runs."))

process_mcmc_round(experiment_dir, round, mcmc_ids, 
                   rhat_threshold=rhat_threshold,
                   min_itr_threshold=min_itr_threshold)


samp_list <- try(readRDS(file.path(mcmc_dir, "mean", "em_fwd", 
                                   paste0("em_", 3),
                                   paste0("mcmc_", 3),
                                   "mcmc_samp.rds")))

samp_dt <- samp_list$samp













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








