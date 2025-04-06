#
# run_approx_mcmc.r
#
# Given an inverse problem object and llikEmulator object, runs various MCMC
# algorithms seeking to characterize the approximate posterior. An experiment 
# tag is passed in as a commandline argument, which is used to locate the 
# experiment output directory. A file named "mcmc_approx_settings.rds" is 
# required to be in this directory, which contains a list of MCMC settings 
# lists and thus controls which MCMC algorithms are run. Moreover, the
# file `<experiment_tag>/inv_prob_setup/inv_prob_list.rds` is required.
#
# The commandline argument `em_dir` gives the path, relative to 
# `<experiment_tag>` where the llikEmulator objects are saved. It is assumed 
# that each subdirectory within `<em_dir>` pertains to a single llikEmulator
# object. The `em_id` commandline argument gives the name of the subdirectory 
# to utilize in the execution of this file. The llikEmulator object is then 
# searched for at the path:
# gp-calibration/output/gp_inv_prob/<experiment_dir>/<em_dir>/<em_id>/em_llik.rds
# Note that `<em_dir>` may contain a sub-path; e.g., "init_emulator/LHS_250".
# Outputs are saved to the directory:
# gp-calibration/output/gp_inv_prob/<experiment_dir>/<run_id>/<em_dir>/<em_id>/
#
# This script is set up to be run remotely on the cluster. The intention is to 
# run this script on each llikEmulator object resulting from a replicate
# design.
#
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)
library(docopt)

# -----------------------------------------------------------------------------
# docopt string for parsing command line arguments.  
# -----------------------------------------------------------------------------

"Usage:
  test_docopt.r [options]
  test_docopt.r (-h | --help)

Options:
  -h --help                                 Show this screen.
  --experiment_tag=<experiment_tag>         Used to locate the base output directory.
  --run_id=<run_id>                         Used to define output directory.
  --em_dir=<out_dir>                        llikEmulator base directory.
  --em_id=<em_id>                           llikEmulator sub-directory.
" -> doc

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Base directory: all paths are relative to this directory.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Directory to source code.
src_dir <- file.path(base_dir, "src")

# Read command line arguments.
cmd_args <- docopt(doc)
experiment_tag <- cmd_args$experiment_tag
run_id <- cmd_args$run_id
em_dir <- cmd_args$em_dir
em_id <- cmd_args$em_id

print(paste0("experiment_tag: ", experiment_tag))
print(paste0("run_id: ", run_id))
print(paste0("em_dir: ", em_dir))
print(paste0("em_id: ", em_id))

# Define directories and ensure required paths exist.
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
em_id_dir <- file.path(experiment_dir, em_dir, em_id)
out_dir <- file.path(experiment_dir, run_id, em_dir, em_id)

print(paste0("experiment_dir: ", experiment_dir))
print(paste0("em_id_dir: ", em_id_dir))
print(paste0("out_dir: ", out_dir))

mcmc_settings_path <- file.path(experiment_dir, "mcmc_approx_settings.rds")
inv_prob_path <- file.path(experiment_dir, "inv_prob_setup", "inv_prob_list.rds")
llik_em_path <- file.path(em_id_dir, "em_llik.rds")
design_path <- file.path(em_id_dir, "design_info.rds")

print("-----> Checking required files exist:")
assert_that(file.exists(mcmc_settings_path))
assert_that(file.exists(llik_em_path))
assert_that(file.exists(inv_prob_path))
assert_that(file.exists(design_path))

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

# Load R project. 
renv::load(base_dir)
print(".libPaths()")
print(.libPaths())
renv::status()

# Create output directory.
dir.create(out_dir, recursive=TRUE)

print("-------------------- Loading files --------------------")

print("-----> MCMC settings")
print(paste0("Loading: ", mcmc_settings_path))
mcmc_settings_list <- readRDS(mcmc_settings_path)

print("-----> Inverse problem object")
print(paste0("Loading: ", inv_prob_path))
inv_prob <- readRDS(inv_prob_path)

print("-----> Log-likelihood emulator")
print(paste0("Loading: ", llik_em_path))
llik_em <- readRDS(llik_em_path)

print("-----> Design info")
print(paste0("Loading: ", design_path))
design_info <- readRDS(design_path)

print("-------------------- Running MCMC --------------------")

# print("-----> Truncating prior based on design point bounds:")
par_prior <- inv_prob$par_prior
# par_prior$bound_lower <- par_prior$param1
# par_prior$bound_upper <- par_prior$param2
# par_prior_trunc <- truncate_prior(par_prior, design_info$bounds)
# print(par_prior_trunc)

print("-----> Setting initial proposal covariance:")
cov_prop_init <- cov(design_info$input)
print("Initial proposal covariance:")
print(cov_prop_init)

# Algorithms using BayesianTools wrapper use their own form of adaptation, 
# so only set proposal covariance for non-BayesianTools algorithms.
for(i in seq_along(mcmc_settings_list)) {
  if(mcmc_settings_list[[i]]$mcmc_func_name != "mcmc_bt_wrapper") {
    mcmc_settings_list[[i]]$cov_prop <- cov_prop_init
  }
}

print("Calling `run_mcmc_comparison()`:")
run_mcmc_comparison(llik_em, par_prior, mcmc_settings_list, 
                    save_dir=out_dir, return=FALSE)










