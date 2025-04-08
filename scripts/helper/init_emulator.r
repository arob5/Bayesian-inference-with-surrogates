#
# init_emulator.r
# This script is intended to be called by `init_emulator.sh`, which is in turn
# intended to be called by the runner script `run_init_emulator.r`. Currently
# this workflow is set up so that each (design tag, emulator tag) combination
# is handled by a separate remote node on the cluster. This means that all 
# design replicates are fit for the (design tag, emulator tag) combination 
# on a single node. Currently this is fine, but if the model fitting process
# becomes more expensive, then it will be more efficient to batch out the 
# design IDs across different nodes as well. Note that this is the initial 
# emulator fitting round, so this file is only used for "round 1".
#
# Fit emulators are saved to 
# experiments/<experiment_tag>/round1/em/<em_tag>/<em_id>/em_llik.rds
# Recall that emulator IDs are defined to be unique within each emulator tag.
# A file `id_map.csv` is saved to the <em_tag> dir that contains the columns
# "em_tag", "em_id", "design_tag", "design_id", "seed". This provides 
# information on which design was used to fit a specific emulator, as well as 
# the random seed that was used in the fitting.
#
# It is assumed that the `id_map.csv` file in the `<em_tag>` directory has
# already been saved. This file is used to identify the design IDs to run, 
# and the em_ids determine where the outputs are saved. Note that `id_map.csv`
# is saved by the runner script `run_init_emulator.r`.
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
  --experiment_tag=<experiment_tag>         The experiment tag.
  --em_tag=<em_tag>                         The emulator tag.
  --design_tag=<design_tag>                 The initial design tag.
  --seed=<seed>                             Seed for random number generator.
" -> doc

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Read command line arguments.
cmd_args <- docopt(doc)
experiment_tag <- cmd_args$experiment_tag
em_tag <- cmd_args$em_tag
design_tag <- cmd_args$design_tag
seed <- cmd_args$seed

set.seed(seed)

print("--------------------Running `init_emulator.r` --------------------")
print(paste0("Experiment tag: ", experiment_tag))
print(paste0("Emulator tag: ", em_tag))
print(paste0("Design tag: ", design_tag))
print(paste0("Seed: ", seed))

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
src_dir <- file.path(code_dir, "src")
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
setup_dir <- file.path(experiment_dir, "output", "inv_prob_setup")
design_dir <- file.path(experiment_dir, "output", "round1", "design", design_tag)
base_out_dir <- file.path(experiment_dir, "output", "round1", "em", em_tag)
em_settings_path <- file.path(experiment_dir, "output", "alg_settings", "em_settings.rds")
design_ids_path <- file.path(experiment_dir, "output", "round1", "design", 
                             design_tag, "id_map.csv")

print(paste0("Using emulator settings: ", em_settings_path))
em_settings <- readRDS(em_settings_path)

print(paste0("Using design ids: ", design_ids_path))
design_ids <- fread(design_ids_path)

print(paste0("Creating output directory: ", base_out_dir))
dir.create(base_out_dir, recursive=TRUE)

if(anyDuplicated(design_ids$design_id)) {
  stop("Found duplicate design IDs. Design ID should be unique within the design tag.")
}

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

# Log-likelihood bounds.
llik_bounds <- inv_prob$llik_obj$get_llik_bounds()


# ------------------------------------------------------------------------------
# Fit emulators, and save results to file.
# ------------------------------------------------------------------------------

print("-------------- Fitting emulators --------------")

for(id in design_ids$design_id) {
  
  # Locate the correct design.
  print(paste0("-----> Design ID: ", id))
  design_path <- file.path(design_dir, paste0("design_", id), "design_info.rds")
  print(paste0("Loading design: ", design_path))
  design_info <- readRDS(design_path)
  
  # Create output directory.
  
  
  
  for(tag in em_tags) {
    
    # Create output directory.
    print(paste0("Emulator tag: ", tag))
    out_dir <- file.path(base_out_dir, id, tag)
    print(paste0("Creating output directory: ", out_dir))
    dir.create(out_dir, recursive=TRUE)
    
    # Fit and save emulator.
    em_llik <- em_list[[tag]]$fit_em(design_info)
    saveRDS(em_llik, file=file.path(out_dir, "em_llik.rds"))
    
    # Compute emulator predictions at test points and save to file.
    test_llik_em(em_llik, out_dir)
  }
}

