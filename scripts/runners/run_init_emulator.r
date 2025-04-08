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

# Required paths.
experiment_dir <- file.path(base_dir, experiment_tag)
src_dir <- file.path(code_dir, "src")
design_dir <- file.path(experiment_dir, "output", "round1", "design")
bash_path <- file.path(base_dir, "scripts", "bash", "run_init_emulator.sh")





















# If not explicitly specified, select all emulators/designs found in 
# design directory.
if(is.null(design_ids)) {
  design_ids <- list.files(design_dir)
}
design_ids <- setdiff(design_ids, "design_settings.rds")

print(paste0("Number of design IDs: ", length(design_ids)))

# Remove the ".rds" from the strings to extract the IDs.
for(i in seq_along(design_ids)) {
  design_ids[i] <- strsplit(design_ids[i], split=".", fixed=TRUE)[[1]][1]
}

# Execute one job per design ID.
print("-----> Starting jobs with design_ids:")
base_cmd <- paste("qsub", bash_path, experiment_tag, design_tag)

for(id in design_ids) {
  print(id)
  cmd <- paste(base_cmd, id)
  system(cmd)
}










