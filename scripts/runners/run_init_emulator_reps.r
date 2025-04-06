#
# run_init_emulator_reps.r
# A wrapper around `run_init_emulator.r` that executes `init_emulator.sh`
# once per design ID that is found. A separate remote job is created for 
# each design ID. By default, runs for all design IDs 
# found within the specified directory, but also allows for specific files to 
# be targeted. 
# 
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)
library(docopt)

# Settings.
experiment_tag <- "vsem"
design_tag <- "LHS_200"

# Specify the design IDs to run. Setting NULL will default to all IDs found
# within the design directory.
design_ids <- NULL

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
design_dir <- file.path(experiment_dir, "init_design", design_tag)
bash_path <- file.path(base_dir, "scripts", "gp_post_approx_paper",
                       "sim_study", "bash_files", "run_init_emulator.sh")

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










