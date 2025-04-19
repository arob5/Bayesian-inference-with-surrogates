#
# run_analyze_mcmc.r
# This script is intended to be run after `run_approx_mcmc.r` and 
# `run_postprocess_mcmc.r` to provide summaries of the approximate MCMC 
# output and compare the various approximate samples to the exact MCMC samples.
# These plots summarize the performance of an MCMC algorithm (i.e., MCMC tag)
# for each emulator tag/design tag combination across the set of initial 
# design replicates.
#
# Andrew Roberts
#

# TODO: 
# Need to update the helper functions used here to also work for rounds beyond
# round one. It seems that I'd want a function that takes as inputs at minimum an 
# initial design tag, and then the code propagates forward through the rounds
# to grab all output stemming from that initial design. Could optionally also
# restrict to particular em_tags, mcmc_tags, or acq_tags.

library(data.table)
library(assertthat)
library(docopt)

# Plotting.
library(ggplot2)
library(patchwork)
library(scales)
library(grid)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Paths
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Experiment and round.
experiment_tag <- "banana"
round <- 1L

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

round_name <- paste0("round", round)

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
mcmc_dir <- file.path(experiment_dir, "output", round_name, "mcmc")
inv_prob_dir <- file.path(experiment_dir, "output", "inv_prob_setup")

# Paths.
mcmc_ids_path <- file.path(mcmc_dir, "id_map.csv")
mcmc_summary_path <- file.path(mcmc_dir, "mcmc_summary.csv")
mcmc_group_info_path <- file.path(mcmc_dir, "group_info.csv")

# Source required scripts.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(base_dir, "scripts", "helper", "sim_study_functions.r"))

# ------------------------------------------------------------------------------
# Load exact MCMC baseline.
# ------------------------------------------------------------------------------

inv_prob <- readRDS(file.path(inv_prob_dir, "inv_prob_list.rds"))
samp_exact <- fread(file.path(inv_prob_dir, "samp_exact.csv"))
samp_exact_stats_uni <- readRDS(file.path(inv_prob_dir, "mcmc_exact_stats_univariate.rds"))
samp_exact_stats_multi <- readRDS(file.path(inv_prob_dir, "mcmc_exact_stats_multivariate.rds"))

# ------------------------------------------------------------------------------
# Read and compile approximate MCMC output.  
# ------------------------------------------------------------------------------

# Extract MCMC IDs.
design_tag_list <- list(round1=c("simple_5"))
em_tag_list <- list(round1=c("em_fwd"))
mcmc_tag_list <- list(round1=c("mean", "marginal"))

mcmc_ids <- get_ids_by_tag(experiment_dir, round=round, 
                           design_tag_list=design_tag_list, 
                           em_tag_list=em_tag_list,
                           mcmc_tag_list=mcmc_tag_list)


# Questions:
#  - Why is pm-ind failing for em_fwd, but not em_llik?
#  - Why is pm-joint failing on every run? (not a priority question)
#  - Why are the pm-joint-rect/pm-ind-rect failing for em_llik but not em_fwd?


# Load MCMC tags. Print basic summary of MCMC runs.
mcmc_summary <- fread(file.path(mcmc_dir, "mcmc_summary.csv"))

mcmc_summary[, .N, by=status]

mcmc_summary[status != "valid", .N, by="mcmc_tag"]
mcmc_summary[status != "valid", .N, by=c("mcmc_tag", "em_tag")]

# Group by mcmc_tag, em_tag. In round 1 this should identify each algorithm 
# so that we can identify how many replicates there are per group.
# TODO: generalize to future round; think I also need to group by em_id.
# Better yet, need a way to just input tags (e.g., initial design tag, em_tag,
# mcmc_tag) and then have a function that loops over the rounds until the 
# current round and identifies all MCMC runs associated with those tags.
mcmc_summary[status=="valid", .(n_valid_reps=.N), by=c("mcmc_tag", "em_tag")]



mcmc_summary <- mcmc_summary[status=="valid", .(n_valid_reps=.N), by=mcmc_tag]
mcmc_tags <- mcmc_summary[n_valid_reps > 1L, mcmc_tag]

# Load statistics of aggregated data.
mcmc_tag_list <- vector(mode="list", length=length(mcmc_tags))
names(mcmc_tag_list) <- mcmc_tags


#
# TODO: need to add multivariate stats to list returned by `get_samp_dt_reps_agg`
#       as well.
#

interval_probs <- seq(.1,1,.1)

print("-----> Loading MCMC tags:")
for(tag in mcmc_tags) {
  print(tag)
  mcmc_tag_list[[tag]] <- get_samp_dt_reps_agg(experiment_dir, round, tag, 
                                               em_tag, design_tag, 
                                               only_valid=TRUE, format_long=FALSE,
                                               interval_probs=interval_probs)
}


# ------------------------------------------------------------------------------
# 1d Marginal Coverage Plots
# ------------------------------------------------------------------------------

plt_dir <- file.path(mcmc_dir, "summary_files", "plots")

# All unique combinations of test_label (i.e., MCMC tag), param_type, and
# param_name.
param_summary <- fread(file.path(mcmc_dir, "summary_files", "param_summary.csv"))
plt_id_vars <- unique(param_summary, by=c("mcmc_tag", "param_type", "param_name"))

# Functions to summarize percentiles of coverage dist over replications.
summary_funcs <- list(median = median,
                      mean = mean,
                      q_lower = function(x) quantile(x, .1),
                      q_upper = function(x) quantile(x, .9))

coverage_plt_list <- list()

for(tag in mcmc_tags) {
  print(paste0("MCMC Tag: ", tag))
  
  coverage_plt_list[[tag]] <- list()
  tag_pars <- plt_id_vars[mcmc_tag==tag]
  tag_coverage_stats <- mcmc_tag_list[[tag]]$cred_intervals
  
  for(i in 1:nrow(tag_pars)) {
    p_type <- tag_pars[i, param_type]
    p_name <- tag_pars[i, param_name]
    cat("\n", p_type, "-", p_name, "\n")
    
    tag_coverage_param <- tag_coverage_stats[(param_type==p_type) & (param_name==p_name)]
    samp_exact_param <- samp_exact[(param_type==p_type) & (param_name==p_name), sample]
    n_exact_samp <- length(samp_exact_param)
    compute_actual_coverage <- function(l, u) sum((samp_exact_param >= l) & (samp_exact_param <= u)) / n_exact_samp
    tag_coverage_param[, actual_coverage := compute_actual_coverage(lower, upper), 
                        by=seq_len(nrow(tag_coverage_param))]
    
    coverage_plt_data <- agg_dt_by_func_list(tag_coverage_param, 
                                             value_col="actual_coverage",
                                             group_cols=c("test_label", "param_type", "param_name", "prob"),
                                             agg_funcs=summary_funcs)
    coverage_plt_data[, nominal_coverage := as.numeric(prob)/100]
    
    plt <- ggplot(coverage_plt_data, aes(x=nominal_coverage)) +
            geom_ribbon(aes(ymin=q_lower, ymax=q_upper), fill="skyblue", alpha=0.4) +
            geom_line(aes(y=median), color="darkblue") +
            geom_line(aes(y=mean), color="darkorange") + 
            geom_abline(slope=1, intercept=0, color="red", linetype="dashed") +
            xlab("Nominal Coverage") + ylab("Actual Coverage") +
            ggtitle(paste(tag, p_name, sep=" - "))
    
    # NOTE: there is currently only one param type so I am ignoring the type
    # here for simplicitly. When generalizing to multiple types need to account
    # for this to avoid duplicate parameter names across types.
    coverage_plt_list[[tag]][[p_name]] <- plt
  }
}

# Save to file.
saveRDS(coverage_plt_list, file.path(mcmc_dir, "summary_files", "coverage_plt_list.rds"))


plot_grid <- function(plt_list, mcmc_tags, par_names, interval_probs) {
  grid_plt_list <- list()
  q_min <- min(interval_probs)
  q_max <- max(interval_probs)
  
  for(i in seq_along(mcmc_tags)) {
    tag <- mcmc_tags[i]
    
    for(j in seq_along(par_names)) {
      par <- par_names[j]
      plt <- plt_list[[tag]][[par]] + 
        # Print axis labels as percentages but exclude printing the "%" sign.
        scale_x_continuous(labels = function(x) format(x * 100, nsmall=0)) +
        scale_y_continuous(labels = function(x) format(x * 100, nsmall=0)) +
        theme_minimal() + 
        theme(axis.title.x = element_blank(),
              plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")) +
        ggtitle(NULL)
      
      # Only add titles to the first row.
      if(i == 1L) {
        # TODO: TEMP
        if(par == "KEXT") nm <- "theta 1"
        if(par == "GAMMA") nm <- "theta 2"
        if(par == "Cv") nm <- "theta 3"
        plt <- plt + ggtitle(nm)
        
        # plt <- plt + ggtitle(par)
      }
      
      # Only add x-axis tick labels to the last row.
      if(i != length(mcmc_tags)) {
        plt <- plt + theme(axis.text.x = element_blank())
      } else {
        plt <- plt + theme(axis.text.x = element_text(size=12))
      }
      
      # Only add y-axis labels and tick labels to the first column.
      if(j == 1L) {
        # TODO: TEMP
        if(tag == "mean-rect") nm <- "mean"
        if(tag == "marginal-rect") nm <- "marginal"
        if(tag == "mcwmh-joint-rect") nm <- "noisy"
        plt <- plt + ylab(nm) + 
                theme(axis.title.y=element_text(size=16),
                      axis.text.y=element_text(size=12))
        
        # plt <- plt + ylab(tag)
      } else {
        plt <- plt + theme(axis.text.y = element_blank(),
                           axis.title.y = element_blank())
      }
      
      # Use coord_cartesian to make sure the plot is square.
      plt <- plt + coord_fixed(ratio = 1) # coord_cartesian(xlim=c(q_min,q_max), ylim=c(q_min,q_max)) +
            
      # Append plot to list.
      grid_plt_list <- c(grid_plt_list, list(plt))
    }
  }
  
  # Format in grid.
  wrap_plots(grid_plt_list, ncol=length(par_names), nrow=length(mcmc_tags))
}


# Plug-in Mean.
tags_mean <- c("mean-rect", "mcwmh-joint-rect")
plt_grid_mean <- plot_grid(coverage_plt_list, tags_mean, par_names=c("KEXT","GAMMA","Cv"),
                           interval_probs) & 
                 theme(plot.background=element_rect(fill="transparent", color=NA))
plot(plt_grid_mean)
ggsave(file.path(plt_dir, "coverage_plots_for_imsi_poster.png"), plt_grid_mean,
       width=7, height=7)



# Comparing mean-rect, marginal-rect, and mcwmh-ind-rect.
tags_rect <- c("marginal-rect", "mcwmh-ind-rect")
plt_grid_rect <- plot_grid(coverage_plt_list, tags_rect, par_names,
                           interval_probs)
plot(plt_grid_rect)
ggsave(file.path(plt_dir, "marg_vs_ind_noisy_rect.png"), plt_grid_rect,
       width=7, height=7)

tags_rect2 <- c("mean-rect", "marginal-rect")
plt_grid_rect2 <- plot_grid_test(coverage_plt_list, tags_rect2, par_names,
                                 interval_probs)
ggsave(file.path(plt_dir, "mean_vs_marg_rect.png"), plt_grid_rect2,
       width=7, height=7)

# TODO: check to make sure these values aren't exactly the same. Need to check
# that joint sampling was actually used here.
tags_noisy <- c("mcwmh-ind-rect", "mcwmh-joint-rect")
plt_grid_noisy <- plot_grid_test(coverage_plt_list, tags_noisy, par_names,
                                 interval_probs)
plot(plt_grid_noisy)
ggsave(file.path(plt_dir, "noisy_ind_vs_joint_rect.png"), plt_grid_noisy,
       width=7, height=7)


tags_noisy2 <- c("mcwmh-joint-rect", "mcwmh-joint-trunc")
plt_grid_noisy2 <- plot_grid_test(coverage_plt_list, tags_noisy2, par_names,
                                  interval_probs)
plot(plt_grid_noisy2)
ggsave(file.path(plt_dir, "noisy_rect_vs_trunc.png"), plt_grid_noisy2,
       width=7, height=7)

# Pseudomarginal approach to sampling marginal dist.
tags_pm_marg <- c("marginal-rect", "pm-ind-rect")
plt_grid_pm_marg <- plot_grid_test(coverage_plt_list, tags_pm_marg, par_names,
                                   interval_probs)
plot(plt_grid_pm_marg)
ggsave(file.path(plt_dir, "pm_vs_marg_rect.png"), plt_grid_pm_marg,
       width=7, height=7)

# Pseudomarginal ind vs joint.
tags_pm <- c("pm-ind-rect", "pm-joint-rect")
plt_grid_pm <- plot_grid_test(coverage_plt_list, tags_pm, par_names,
                              interval_probs)
plot(plt_grid_pm)
ggsave(file.path(plt_dir, "pm_ind_vs_joint.png"), plt_grid_pm,
       width=7, height=7)




# TODO:
# Add labels, and then use wrap_plots to consolidate the legend into one.
# Need to make plot axes identical across all plots; currently, the grid lines
# are not the same across all plots.












