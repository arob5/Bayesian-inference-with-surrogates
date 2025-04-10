#
# approx_mcmc.r
# Runs the MCMC step in the simulation study workflow (this script is used
# for all rounds, including round one). Each MCMC run is given an ID which 
# is unique within the round/MCMC tag. Recall that the MCMC ID requires 
# an emulator tag/emulator ID as input to select the emulator that will 
# be used within approximate MCMC. This script is set up to run for a specific
# experiment, round, emulator tag, emulator ID, and a set of MCMC tags. The
# first four identifiers uniquely point to a specific fit emulator model that
# will be used in the MCMC. One MCMC run will be produced per MCMC tag.
# All of this information is supplied by command line arguments. 
#
# The script is intended to be called by approx_mcmc.sh, which is in turn
# called by run_approx_mcmc.r. Outputs are saved to:
# experiments/<experiment_tag>/output/round<round>/mcmc/<mcmc_tag>/<em_tag>/em_<em_id>/mcmc_<mcmc_id>/samp.rds
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
  run_approx_mcmc.r [options]
  run_approx_mcmc.r (-h | --help)

Options:
  -h --help                                 Show this screen.
  --experiment_tag=<experiment_tag>         The experiment tag.
  --round=<round>                           Round number (integer)
  --em_tag=<em_tag>                         The emulator tag (string)
  --em_id=<em_id>                           Emulator ID (integer)
  --mcmc_tags=<mcmc_tags>                   Comma-separated list of MCMC tags (no spaces)
" -> doc

# Read command line arguments.
cmd_args <- docopt(doc)
experiment_tag <- cmd_args$experiment_tag
round <- as.integer(cmd_args$round)
em_tag <- cmd_args$em_tag
em_id <- as.integer(cmd_args$em_id)
mcmc_tags <- cmd_args$mcmc_tags

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

print("--------------------Running `approx_mcmc.r` --------------------")
print(paste0("Experiment tag: ", experiment_tag))
print(paste0("Round: ", round))
print(paste0("Emulator tag: ", em_tag))
print(paste0("Emulator ID: ", em_id))
print(paste0("MCMC tags: ", mcmc_tags))

# Convert comma-separated list to vector.
mcmc_tags <- strsplit(mcmc_tags, ",", fixed=TRUE)[[1]]

# String versions of IDs.
round_name <- paste0("round", round)
em_id_name <- paste0("em_", em_id)

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
src_dir <- file.path(code_dir, "src")
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
setup_dir <- file.path(experiment_dir, "output", "inv_prob_setup")
mcmc_dir <- file.path(experiment_dir, "output", round_name, "mcmc")

print(paste0("Creating MCMC directory: ", mcmc_dir))
dir.create(mcmc_dir, recursive=TRUE)

# Source required files.
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
source(file.path(src_dir, "basis_function_emulation.r"))

# Load inverse problem setup information.
inv_prob <- readRDS(file.path(setup_dir, "inv_prob_list.rds"))

# ------------------------------------------------------------------------------
# Load MCMC settings 
# ------------------------------------------------------------------------------

mcmc_settings_path <- file.path(experiment_dir, "output", 
                                "alg_settings", "mcmc_settings.rds")
print(paste0("Reading MCMC settings: ", mcmc_settings_path))
mcmc_settings <- readRDS(mcmc_settings_path)
valid_mcmc_tags <- sapply(mcmc_settings, function(x) x$test_label)
invalid_tags <- setdiff(mcmc_tags, valid_mcmc_tags)

if(length(invalid_tags) > 0L) {
  stop("MCMC tags not found in emulator settings: ", paste0(invalid_tags, collapse=", "))
}

mcmc_settings <- mcmc_settings[mcmc_tags]

# ------------------------------------------------------------------------------
# Load fit emulator model. 
# ------------------------------------------------------------------------------

# Read ID map.
em_id_path <- file.path(experiment_dir, "output", round_name, "em", "id_map.csv")
print(paste0("Reading emulator IDs from: ", em_id_path))
em_ids <- fread(em_id_path)

# Select the design tag/design ID for the emulator ID, which is used to locate
# the path of the emulator.
e_tag <- em_tag
e_id <- em_id
em_ids <- em_ids[(em_tag==e_tag) & (em_id==e_id)]
if(nrow(em_ids) != 1L) {
  stop("Subsetting the emulator ID map should select exactly one row.")
}

design_tag <- em_ids$design_tag
design_id_name <- paste0("design_", em_ids$design_id)

# Load emulator
em_path <- file.path(experiment_dir, "output", round_name, "em", em_tag,
                     design_tag, design_id_name, em_id_name, "em_llik.rds")
print(paste0("Reading emulator from: ", em_path))
llik_em <- readRDS(em_path)

# ------------------------------------------------------------------------------
# Run MCMC ID map.
# ------------------------------------------------------------------------------

mcmc_ids_path <- file.path(mcmc_dir, "id_map.csv")
print(paste0("Reading MCMC ID map: ", mcmc_ids_path))
mcmc_ids <- fread(mcmc_ids_path)

# TODO: left off here.



# ------------------------------------------------------------------------------
# Run MCMC 
# ------------------------------------------------------------------------------

print("-------------------- Running MCMC --------------------")

# Prior distribution.
par_prior <- inv_prob$par_prior

# Setting initial proposal covariance based on prior.
print("-----> Setting initial proposal covariance:")
cov_prop_init <- cov(llik_em$get_design_inputs())
print("Initial proposal covariance:")
print(cov_prop_init)

# Algorithms using BayesianTools wrapper use their own form of adaptation, 
# so only set proposal covariance for non-BayesianTools algorithms.
for(i in seq_along(mcmc_settings)) {
  if(mcmc_settings[[i]]$mcmc_func_name != "mcmc_bt_wrapper") {
    mcmc_settings[[i]]$cov_prop <- cov_prop_init
  }
}

# Define output directories.
out_dirs <- file.path(mcmc_dir, mcmc_tags, em_tag, em_id, mcmc_ids)
print("Output directories:")
for(dir in out_dirs) print(dir)

print("Calling `run_mcmc_comparison()`:")
run_mcmc_comparison(llik_em, par_prior, mcmc_settings, 
                    save_dir=out_dir, return=FALSE)










