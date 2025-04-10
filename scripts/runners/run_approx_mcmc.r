#
# run_approx_mcmc.r
# Runs the MCMC step in the simulation study workflow (this script is used
# for all rounds, including round one). Currently, the user specifies an 
# experiment tag, round, set of emulator tags, and MCMC tags, and this script 
# will dispatch jobs to run each MCMC algorithm on each emulator ID within the 
# specified emulator tag. This could be expanded to provide more control; e.g., 
# control which algs are run for different emulators tags; or only choose 
# emulators associated with a certain initial design tag.
#
# MCMC outputs are saved to:
# experiments/<experiment_tag>/output/round_<round>/mcmc/<mcmc_tag>/<em_tag>/em_<em_id>/em_llik.rds
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

# Determine emulators and designs to use. If `mcmc_tags` is NULL, will run
# all MCMC algorithms.
em_tags <- c("em_llik", "em_fwd")
mcmc_tags <- NULL

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

round_name <- paste0("round", round)

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
em_dir <- file.path(experiment_dir, "output", round_name, "em")
mcmc_dir <- file.path(experiment_dir, "output", round_name, "mcmc")
bash_path <- file.path(base_dir, "scripts", "bash", "approx_mcmc.sh")

# Required input files.
em_settings_path <- file.path(experiment_dir, "output", "alg_settings", "em_settings.rds")

# ------------------------------------------------------------------------------
# Ensure MCMC tags are valid
# ------------------------------------------------------------------------------

mcmc_settings_path <- file.path(experiment_dir, "output", 
                                "alg_settings", "mcmc_settings.rds")
print(paste0("Reading MCMC settings: ", mcmc_settings_path))
mcmc_settings <- readRDS(mcmc_settings_path)
valid_mcmc_tags <- sapply(mcmc_settings, function(x) x$test_label)

if(is.null(mcmc_tags)) {
  mcmc_tags <- valid_mcmc_tags
} else {
  invalid_tags <- setdiff(mcmc_tags, valid_mcmc_tags)
  
  if(length(invalid_tags) > 0L) {
    stop("MCMC tags not found in emulator settings: ", paste0(invalid_tags, collapse=", "))
  }
}


# ------------------------------------------------------------------------------
# Create new MCMC IDs and save ID map file.
# ------------------------------------------------------------------------------

# Read emulator ID map.
em_ids_path <- file.path(em_dir, "id_map.csv")
print(paste0("Reading emulator ID map: ", em_ids_path))
em_ids <- fread(em_ids_path)[, .(em_tag, em_id)]

invalid_em_tags <- setdiff(em_tags, unique(em_ids$em_tag))
if(length(invalid_em_tags) > 0L) {
  stop("Em tags not found in emulator ID map file: ", paste0(invalid_em_tags, collapse=", "))
}

em_ids <- em_ids[em_tag %in% em_tags]

# Create MCMC IDs/ID map.
mcmc_id_map <- data.table(mcmc_id=integer(), mcmc_tag=character(), 
                          em_tag=character(), em_id=integer(), seed=integer())

for(mcmc_tag in mcmc_tags) {
  id_map_tag <- copy(em_ids)
  seeds <- sample.int(n=.Machine$integer.max, size=nrow(id_map_tag))
  id_map_tag[, `:=`(mcmc_id=1:.N, mcmc_tag=mcmc_tag, seed=seeds)]
  mcmc_id_map <- rbindlist(list(mcmc_id_map, id_map_tag), use.names=TRUE)
}

# Save to file.
print(paste0("Creating MCMC directory: ", mcmc_dir))
dir.create(mcmc_dir, recursive=TRUE)

mcmc_id_map_path <- file.path(mcmc_dir, "id_map.csv")
print(paste0("Saving MCMC ID map: ", mcmc_id_map_path))
fwrite(mcmc_id_map, mcmc_id_map_path)

# ------------------------------------------------------------------------------
# Batch out jobs.
#   Currently sending each em_tag/em_id combo to a node. For more expensive 
#   MCMC algorithms, will probably want to also split the MCMC algorithms
#   across multiple nodes, even with a single em_id.
# ------------------------------------------------------------------------------

# Batch by em_tag/em_id.
batch_ids <- unique(mcmc_id_map[, .(em_tag, em_id)])
print(paste0("Preparing to submit ", nrow(batch_ids), " jobs."))

# Comma-separated format for MCMC tags to pass via commandline argument.
mcmc_tags_str <- paste0(mcmc_tags, collapse=",")

# Start jobs.
print("-----> Starting jobs with em_tag/em_ids:")
base_cmd <- paste("qsub", bash_path, experiment_tag, round)

for(i in 1:nrow(batch_ids)) {
  em_tag <- batch_ids[i, em_tag]
  em_id <- batch_ids[i, em_id]

  print(paste(em_tag, em_id, sep=" / "))  
  cmd <- paste(base_cmd, em_tag, em_id, mcmc_tags_str)
  system(cmd)
}












# Base directory: all paths are relative to this directory.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
bash_path <- file.path(base_dir, "scripts", "gp_post_approx_paper", 
                       "sim_study", "bash_files", "run_approx_mcmc.sh")

# Directory to source code.
src_dir <- file.path(base_dir, "src")

# Set variables controlling filepaths.
experiment_tag <- "vsem"
run_id <- "mcmc_round1"
em_dir <- "init_emulator/LHS_200"

# Specify specific emulator/design IDs. If NULL, will automatically select 
# all found within directory `<experiment_tag>/<em_dir>`.
em_ids <- NULL

output_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag, em_dir)

print(paste0("experiment_tag: ", experiment_tag))
print(paste0("run_id: ", run_id))
print(paste0("em_dir: ", em_dir))
print(paste0("output_dir: ", output_dir))


# ------------------------------------------------------------------------------
# Dispatch job for each emulator/design.
# ------------------------------------------------------------------------------

# If not explicitly specified, select all emulators/designs found in 
# directory.
if(is.null(em_ids)) {
  em_ids <- list.files(output_dir)
  em_id_dir_sel <- !grepl(".o", em_ids)
  em_ids <- em_ids[em_id_dir_sel]
}

print(paste0("Number of em_ids: ", length(em_ids)))

# Start jobs.
print("-----> Starting jobs with em_ids:")
base_cmd <- paste("qsub", bash_path, experiment_tag, run_id, em_dir)

for(em_id in em_ids) {
  print(em_id)
  cmd <- paste(base_cmd, em_id)
  system(cmd)
}




