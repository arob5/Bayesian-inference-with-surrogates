#
# init_design.r
# Part 2 in the inverse problem simulation study workflow, to be run after
# `inv_prob_setup.r`. This script:
# (1) loads inverse problem data saved to file by `inv_prob_setup.r`. 
# (2) Generates multiple random replications of an initial design to train 
#     log-likelihood and forward model emulators.
# (3) The outputs are saved to `<experiment_tag>/init_design/<design_tag>`.
#     Each replicate design is saved to a file within this directory named as
#     `design_info_<seed>.rds`, where `seed` is the seed used for the random 
#     number generation in creating the design. In addition to these files, 
#     an additional file called `design_settings.rds` is saved that stores the 
#     settings used when running this file.
#
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)
library(docopt)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Used to identify the location of the experiment, which has already been 
# initiated by `inv_prob_setup.r`.
experiment_tag <- "vsem"

# Design methods settings.
n_design <- 200L
design_method <- "LHS"

# Initial design tag.
design_tag <- paste(design_method, n_design, sep="_")

# Number of design replications.
n_rep <- 100L

print("--------------------Running `init_design.r` --------------------")
print(paste0("Experiment tag: ", experiment_tag))
print(paste0("Design tag: ", design_tag))
print(paste0("n_design: ", n_design))
print(paste0("design_method: ", design_method))
print(paste0("n_rep: ", n_rep))

settings <- list(experiment_tag=experiment_tag,
                 design_tag=design_tag,
                 n_design=n_design,
                 design_method=design_method,
                 n_rep=n_rep)

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
src_dir <- file.path(base_dir, "src")
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
setup_dir <- file.path(experiment_dir, "inv_prob_setup")
out_dir <- file.path(experiment_dir, "init_emulator", design_tag)

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

# Load R project. 
renv::load(base_dir)
print(".libPaths()")
print(.libPaths())
renv::status()

# Create output directories.
dir.create(out_dir, recursive=TRUE)

# Load inverse problem setup information.
inv_prob <- readRDS(file.path(setup_dir, "inv_prob_list.rds"))

print(paste0("Output directory: ", out_dir))
print("Saving design settings.")
saveRDS(settings, file.path(out_dir, "design_settings.rds"))

# ------------------------------------------------------------------------------
# Define function that creates a single replicate initial design.
# ------------------------------------------------------------------------------

construct_init_design <- function(seed) {
  print("-------------- Initiating new design --------------")
  set.seed(seed)
  print(paste0("Seed: ", seed))
  
  design_info <- get_init_design_list(inv_prob, design_method, n_design)
  saveRDS(design_info, file=file.path(out_dir, paste0(seed, ".rds")))
}

# ------------------------------------------------------------------------------
# Create replicate designs
# ------------------------------------------------------------------------------

print("-------------- Generating replicate designs --------------")

# Seeds for each replicate design.
seeds <- sample.int(n=.Machine$integer.max, size=settings$n_rep)

# Generate replicate designs.
for(seed in seeds) construct_init_design(seed)



