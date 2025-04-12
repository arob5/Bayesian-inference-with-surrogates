#
# run_approx_mcmc.r
# Runs the MCMC step in the simulation study workflow (this script is used
# for all rounds, including round one). Currently, the user specifies an 
# experiment tag, round, set of emulator tags, and MCMC tags, and this script 
# will dispatch jobs to run each MCMC algorithm on each emulator ID within the 
# specified emulator tag. Allows different sets of MCMC tags to be run for 
# different emulator tags - see below settings section for details.
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

# Determine emulators and designs to use. This is a list, where each name 
# is a em_tag, and each value is a list of mcmc_tags to run for that em_tag.
# Setting a value to NULL will run all MCMC tags.
run_list <- list(
  em_fwd = c("mean", "marginal", "mcwmh-joint", "mcwmh-ind", "pm-joint", "pm-ind"),
  em_llik = NULL
)


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

# Validate MCMC tags and fill in NULL values with vector of all MCMC tags.
for(i in seq_along(run_list)) {
  if(is.null(run_list[[i]])) {
    run_list[[i]] <- valid_mcmc_tags
  } else {
    invalid_tags <- setdiff(run_list[[i]], valid_mcmc_tags)
    
    if(length(invalid_tags) > 0L) {
      stop("MCMC tags not found in emulator settings: ", paste0(invalid_tags, collapse=", "))
    }
  }
}


# ------------------------------------------------------------------------------
# Create new MCMC IDs and save ID map file.
# ------------------------------------------------------------------------------

# Read emulator ID map.
em_ids_path <- file.path(em_dir, "id_map.csv")
print(paste0("Reading emulator ID map: ", em_ids_path))
em_ids <- fread(em_ids_path)[, .(em_tag, em_id)]

em_tags <- names(run_list)
invalid_em_tags <- setdiff(em_tags, unique(em_ids$em_tag))
if(length(invalid_em_tags) > 0L) {
  stop("Em tags not found in emulator ID map file: ", paste0(invalid_em_tags, collapse=", "))
}

em_ids <- em_ids[em_tag %in% em_tags]

# Create MCMC IDs/ID map.
mcmc_id_map <- data.table(mcmc_id=integer(), mcmc_tag=character(), 
                          em_tag=character(), em_id=integer(), seed=integer())

for(e_tag in em_tags) {
  id_map_em_tag <- em_ids[em_tag == e_tag]
  
  for(mcmc_tag in run_list[[e_tag]]) {
    id_map_mcmc_tag <- copy(id_map_em_tag)
    seeds <- sample.int(n=.Machine$integer.max, size=nrow(id_map_mcmc_tag))
    id_map_mcmc_tag[, `:=`(mcmc_id=1:.N, mcmc_tag=mcmc_tag, seed=seeds)]
    mcmc_id_map <- rbindlist(list(mcmc_id_map, id_map_mcmc_tag), use.names=TRUE)
  }
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
#   across multiple nodes, even with a single em_id. The bash script accepts
#   a command line argument to subset the MCMC tags that are run, but for now
#   we just pass "all" to use all MCMC tags found in the MCMC ID map.
# ------------------------------------------------------------------------------

# Batch by em_tag/em_id.
batch_ids <- unique(mcmc_id_map[, .(em_tag, em_id)])
print(paste0("Preparing to submit ", nrow(batch_ids), " jobs."))

# Start jobs.
print("-----> Starting jobs with em_tag/em_ids:")
base_cmd <- paste("qsub", bash_path, experiment_tag, round)

for(i in 1:nrow(batch_ids)) {
  em_tag <- batch_ids[i, em_tag]
  em_id <- batch_ids[i, em_id]

  print(paste(em_tag, em_id, sep=" / "))  
  cmd <- paste(base_cmd, em_tag, em_id, "all")
  system(cmd)
}
