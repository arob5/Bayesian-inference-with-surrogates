#
# run_init_emulator.r
#
# The runner script for the initial emulator fitting step (fitting to the 
# initial designs). This script is thus run once per the experiment, and writes
# output to the `experiments/<experiment_tag>/output/round1/em` directory.
#
# This script works by specifying an experiment tag, and lists of design
# tags and emulator tags. One emulator will be fit per emulator tag per
# design ID within the specified design tag. Currently, jobs are batched
# by unique (emulator tag, design tag) combinations, using the helper files
# `scripts/helper/init_emulator.r` and `scripts/bash/init_emulator.sh`. 
# Emulator IDs (which are unique within each round/emulator tag) are created
# in this script and saved to 
# `experiments/<experiment_tag>/output/round1/em/<em_tag>/id_map.csv`. These
# ID map files provide the information to determine which design was used
# to fit each emulator. They are read in by the helper files mentioned above
# when fitting the emulators.
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

# Experiment
experiment_tag <- "banana"

# Determine emulators and designs to use.
em_tags <- c("em_llik", "em_fwd")
design_tags <- c("simple_5")


# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
design_dir <- file.path(experiment_dir, "output", "round1", "design")
em_dir <- file.path(experiment_dir, "output", "round1", "em")
bash_path <- file.path(base_dir, "scripts", "bash", "init_emulator.sh")

# Required input files.
em_settings_path <- file.path(experiment_dir, "output", "alg_settings", "em_settings.rds")

# Check emulator tags.
print(paste0("Using emulator settings: ", em_settings_path))
em_settings <- readRDS(em_settings_path)
valid_em_tags <- sapply(em_settings, function(x) x$em_label)
invalid_tags <- setdiff(em_tags, valid_em_tags)

if(length(invalid_tags) > 0L) {
  stop("Emulator tags not found in emulator settings: ", paste0(invalid_tags, collapse=", "))
}


# ------------------------------------------------------------------------------
# Create ID map file and write to file. 
# ------------------------------------------------------------------------------

# ID map is saved separately for each tag, but we concatenate them here for use
# below when running the qsub jobs.
id_map <- data.table(design_id=integer(), design_tag=character(),
                     em_tag=character(), em_id=integer(), seed=integer())

for(em_tag in em_tags) {
  print(paste0("-----> Emulator tag: ", em_tag))
  em_id_map <- data.table(design_id=integer(), design_tag=character())
  
  for(design_tag in design_tags) {
    print(paste0("    Design tag: ", design_tag))
    design_ids_path <- file.path(design_dir, design_tag, "id_map.csv")
    print(paste0("    Design IDs path: ", design_ids_path))
    design_ids <- fread(design_ids_path)[, .(design_tag, design_id)]
    em_id_map <- rbindlist(list(em_id_map, design_ids), use.names=TRUE)
  }
  
  seeds <- sample.int(n=.Machine$integer.max, size=nrow(em_id_map))
  em_id_map[, `:=`(em_tag=em_tag, em_id=1:.N, seed=seeds)]
  em_tag_dir <- file.path(em_dir, em_tag)
  dir.create(em_tag_dir)
  em_id_map_path <- file.path(em_tag_dir, "id_map.csv")
  print(paste0("Saving em tag IDs: ", em_id_map_path))
  fwrite(em_id_map, em_id_map_path)
  
  id_map <- rbindlist(list(id_map, em_id_map), use.names=TRUE)
}

# ------------------------------------------------------------------------------
# Create qsub jobs to fit emulators. 
# ------------------------------------------------------------------------------

runs <- unique(id_map[, .(em_tag, design_tag)])

# Execute one job per design ID.
print("-----> Starting jobs:")
base_cmd <- paste("qsub", bash_path, experiment_tag)

for(i in 1:nrow(runs)) {
  em_tag <- runs[i, em_tag]
  design_tag <- runs[i, design_tag]
  print(paste(em_tag, design_tag, sep=" --- "))
  
  cmd <- paste(base_cmd, em_tag, design_tag)
  system(cmd)
}

