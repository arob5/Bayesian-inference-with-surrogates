#
# run_approx_mcmc_reps.r
# A wrapper around `run_approx_mcmc.r` that runs run_approx_mcmc.r` for all 
# designs/emulators within a specified directory. Each run is dispatched 
# to a different remote job using `qsub`. By default, runs for all emulators 
# found within the specified directory, but also allows for specific files to 
# be targeted. 
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




