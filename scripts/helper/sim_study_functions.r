# sim_study_functions.r
# 
# Functions related to conducting a large-scale simulation study aimed at 
# testing methods for posterior approximation and sequential design using 
# surrogate models.
#

# ------------------------------------------------------------------------------
# MCMC Processing
# ------------------------------------------------------------------------------
  
process_mcmc_run <- function(samp_list, rhat_threshold=1.05, 
                             min_itr_threshold=500L, ...) {
  # Post-processing for a single MCMC run (with potentially multiple chains).
  # A chain is defined as "valid" if the within-chain split R-hat values for 
  # all parameters do not exceed `rhat_threshold`. `n_itr` is the total number  
  # of iterations with non-NA values across all valid chains. An MCMC run 
  # (test_label) is marked as valid if it has at least one valid chain and 
  # at least one valid iteration.
  #
  # `samp_list` is a list returned by the MCMC code, with elements "samp", 
  # "info", and "output_list". `min_itr_threshold` is the minimum number of 
  # samples we allow for a single chain. 

  # Check to see if error occurred during run.
  err_occurred <- !is.null(samp_list$output_list[[1]]$condition)
  if(err_occurred) {
    mcmc_summary <- data.table(n_chains=0L, max_rhat=NA, status="mcmc_err")
    return(list(summary=mcmc_summary, chain_info=NULL, rhat=NULL))
  }
  
  if(length(unique(samp_list$samp$test_label)) > 1L) {
    stop("`process_mcmc_run` is defined to operate on a single test label.")
  }
  
  # MCMC samples.
  samp_dt <- samp_list$samp
  
  # Process chain by chain: increase burn-in until each chain individually 
  # satisfies rhat threshold. If a chain fails to meet threshold for any 
  # tried burnin, then `process_mcmc_chain` returns empty data.table.
  chains <- unique(samp_dt$chain_idx)
  chain_list <- lapply(chains, function(i) process_mcmc_chain(samp_dt, i, 
                                                              rhat_threshold=rhat_threshold, 
                                                              min_itr_threshold=min_itr_threshold, ...))
  samp_dt <- rbindlist(chain_list, use.names=TRUE)
  if(nrow(samp_dt) == 0L) {
    mcmc_summary <- data.table(n_chains=0L, max_rhat=NA, status="processing_failed")
    return(list(summary=mcmc_summary, chain_info=NULL, rhat=NULL))
  }
  
  # Parameter level summary: within-chain Rhat statistics.
  id_cols <- c("test_label", "chain_idx", "param_type", "param_name")
  rhat_dt <- calc_R_hat(samp_dt, within_chain=TRUE)$R_hat_vals
  rhat_dt <- rhat_dt[, .SD, .SDcols=c(id_cols, "R_hat")]
  
  # Chain level summary: Maximum within within-chain Rhat over all parameters 
  # within the chain.
  rhat_max_dt <- rhat_dt[, .(rhat=max(R_hat, na.rm=TRUE)), by=.(test_label, chain_idx)]
  other_chain_info <- samp_dt[, .(itr_min=min(itr), itr_max=max(itr)),
                              by=.(test_label, chain_idx)]
  chain_info <- data.table::merge.data.table(rhat_max_dt, other_chain_info,
                                             by=c("test_label", "chain_idx"))

  # Calculate chain weights. Note that any invalid chains have been dropped at 
  # this point, so the weights will sum to one only over the remaining valid
  # chains. First need to subset `info_dt` to align with the adjustments
  # to `samp_dt`.
  info_dt <- samp_list$info
  info_dt <- select_mcmc_samp(info_dt, chain_idcs=chain_info$chain_idx)
  info_list <- list()
  for(i in chain_info$chain_idx) {
    info_list[[i]] <- select_mcmc_samp(info_dt, chain_idcs=i,
                                       itr_start=chain_info[chain_idx==i,itr_min],
                                       itr_stop=chain_info[chain_idx==i,itr_max])
  }
  
  info_dt <- rbindlist(info_list, use.names=TRUE)
  chain_weights <- calc_chain_weights(info_dt)
  chain_info <- data.table::merge.data.table(chain_info, chain_weights,
                                             by=c("test_label", "chain_idx"))
  
  
  # Compute MCMC run summary.
  mcmc_summary <- data.table(n_chains = nrow(chain_info),
                             max_rhat = max(chain_info$rhat),
                             status = "valid")
  
  return(list(summary=mcmc_summary, chain_info=chain_info, rhat=rhat_dt))
}


process_mcmc_chain <- function(samp_dt, chain_idx, rhat_threshold=1.05, 
                               min_itr_threshold=500L, itr_start=1L,
                               n_tries=10L) {
  # `itr_start` can be used to specify an initial burn-in, before computing 
  # any Rhat statistics. `n_tries` is the maximum number of adjustments 
  # that will be made to the burn-in before giving up.

  chain_dt <- select_mcmc_samp(samp_dt, itr_start=itr_start, chain_idcs=chain_idx)
  
  if(nrow(unique(chain_dt[, .(test_label, chain_idx)])) > 1L) {
    stop("`process_mcmc_chain` is defined to operate on a single chain and test label.")
  }
  
  # If the chain passed as argument already doesn't meet the min itr 
  # threshold, return empty data.table.
  if(!chain_meets_min_itr_threshold(chain_dt, min_itr_threshold)) {
    return(get_empty_samp_dt())
  }
  
  # Maximum rhat over all parameters.
  rhat <- calc_R_hat(chain_dt, within_chain=TRUE)$R_hat_vals[,max(R_hat)]
  
  # Compute candidate burn-in values to try sequentially until rhat threshold
  # is met
  itr_cutoffs <- round(seq(chain_dt[,min(itr)], chain_dt[,max(itr)], 
                       length.out=n_tries))
  i <- 2L
  
  # Note that the condition on minimum iteration threshold prevents infinite 
  # loop here, since the final cutoff will be the final iteration and thus 
  # violate the minimum iteration threshold.
  while(isTRUE(rhat > rhat_threshold)) {
    
    # Increase burn-in size.
    chain_dt <- select_mcmc_itr(chain_dt, itr_start=itr_cutoffs[i])
    
    # If the chain doesn't meet the min itr threshold, return empty data.table.
    if(!chain_meets_min_itr_threshold(chain_dt, min_itr_threshold)) {
      return(get_empty_samp_dt())
    }
    
    # Maximum rhat over all parameters.
    rhat <- calc_R_hat(chain_dt, within_chain=TRUE)$R_hat_vals[,max(R_hat)]
    
    i <- i + 1L
  } 
  
  return(chain_dt)
}


chain_meets_min_itr_threshold <- function(chain_dt, min_itr_threshold) {
  if(nrow(unique(chain_dt[, .(test_label, chain_idx)])) > 1L) {
    stop("`process_mcmc_chain` is defined to operate on a single chain and test label.")
  }
  
  n_itr_by_par <- chain_dt[, .N, by=.(param_type, param_name)]
  n_itr_by_par <- unique(n_itr_by_par$N)
  
  if(length(n_itr_by_par) > 1) {
    stop("Different parameters within MCMC chain have different number of iterations.")
  }
  
  n_itr_by_par >= min_itr_threshold
}


# ------------------------------------------------------------------------------
# Loading/Assembling Processed MCMC samples
# ------------------------------------------------------------------------------

get_mcmc_ids <- function(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                         only_valid=TRUE) {
  # Assembles an ID map table with columns "mcmc_id", "em_id", "em_tag",
  # "design_id", and "design_tag" containing the set of MCMC IDs associated
  # with the specified experiment, round, MCMC tag, emulator tag, and design
  # tag specified in the function arguments. If `only_valid` is TRUE, then 
  # the MCMC iterations and chains are subset using the MCMC post-processing
  # results in the "summary_files" directory. This also implies that MCMC runs
  # with no valid chains will be dropped entirely. Setting `only_valid=TRUE`
  # also attaches additional information as columns in the returned data.table,
  # including Rhat values and chain weights.
  #
  # NOTE: 
  # Currently this function returns a data.table that is unique by `mcmc_id`
  # if `only_valid=FALSE` and is unique by (`mcmc_id`, `chain_idx`) if 
  # `only_valid=TRUE`. This is a bit confusing and should probably be changed.
  
  # Assemble set of MCMC tags that are associated with the specified experiment,
  # round, MCMC tag, and emulator tag.
  round_tag <- paste0("round", round)
  mcmc_dir <- file.path(experiment_dir, round_tag, "mcmc", mcmc_tag)
  mcmc_id_map <- fread(file.path(mcmc_dir, "id_map.csv"))
  
  # Subset MCMC IDs to those associated with specified emulator tag.
  em_tag_curr <- em_tag
  mcmc_id_map <- mcmc_id_map[em_tag==em_tag_curr]
  
  # Subset MCMC IDs to those associated with specified design tag.
  design_tag_curr <- design_tag
  em_id_map <- fread(file.path(experiment_dir, round_tag, "em", em_tag, "id_map.csv"))
  mcmc_id_map <- data.table::merge.data.table(mcmc_id_map, em_id_map, by="em_id",
                                              all.x=TRUE, all.y=FALSE)
  mcmc_id_map <- mcmc_id_map[design_tag==design_tag_curr]
  
  # Use MCMC post-processing results to select only valid runs/chains/itrs.
  if(only_valid) {
    
    # Retain only valid runs.
    mcmc_tag_curr <- mcmc_tag
    summary_dir <- file.path(experiment_dir, round_tag, "mcmc", "summary_files")
    run_summary <- fread(file.path(summary_dir, "mcmc_summary.csv"))
    run_summary <- run_summary[(status=="valid") & (mcmc_tag == mcmc_tag_curr)]
    mcmc_id_map <- data.table::merge.data.table(mcmc_id_map, run_summary, 
                                                all=FALSE, by="mcmc_id")
    
    # Retain only valid chains, and set burn-ins determined by preprocessing.
    chain_summary <- fread(file.path(summary_dir, "chain_summary.csv"))
    chain_summary <- chain_summary[mcmc_tag==mcmc_tag_curr]
    mcmc_id_map <- data.table::merge.data.table(mcmc_id_map, chain_summary,
                                                all=FALSE, by=c("mcmc_id", "mcmc_tag"))
  }
  
  return(mcmc_id_map)
}


load_samp_mat <- function(experiment_dir, round, mcmc_tag, mcmc_id, 
                          only_valid=TRUE, n_subsamp=NULL, ...) {
  # Loads MCMC samples from a single run, which is uniquely identified 
  # by (roumd, mcmc_tag, mcmc_id) within the given experiment directory.
  # If `only_valid = TRUE` drops invalid runs/chains/iterations based on the
  # MCMC preprocessing step. If `n_subsamp` is an integer less than the 
  # number of valid samples, will return a subsample of the (valid) MCMC
  # output of size `n_subsamp`.
  #
  # NOTE: call to `select_mcmc_samp_mat()` assumes that the samp_dt contains
  #       only a single param type. 
  #
  # TODO: take into account weights when subsampling.
  
  mcmc_dir <- file.path(experiment_dir, paste0("round", round), "mcmc")
  
  # Load samples.
  samp_dt <- readRDS(file.path(mcmc_dir, mcmc_tag, mcmc_id, "samp.rds"))$samp
  
  # Extract valid samples.
  if(only_valid) {
    chain_summary <- fread(file.path(mcmc_dir, "summary_files", "chain_summary.csv"))
    mcmc_id_curr <- mcmc_id
    mcmc_tag_curr <- mcmc_tag
    chain_summary <- chain_summary[(mcmc_id==mcmc_id_curr) & (mcmc_tag==mcmc_tag_curr),
                                   .(chain_idx, itr_min, itr_max)]
    samp_dt <- select_itr_by_chain(samp_dt, chain_summary)
  }
  
  # Convert to matrix.
  samp_mat <- select_mcmc_samp_mat(samp_dt, ...)
  
  # Sub-sample.
  if(!is.null(n_subsamp) && isTRUE(n_subsamp < nrow(samp_mat))) {
    idcs <- sample(1:nrow(samp_mat), size=n_subsamp, replace=FALSE)
    samp_mat <- samp_mat[idcs,]
  }
  
  return(samp_mat)
}


get_samp_dt_reps <- function(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE) {
  # Loads all MCMC samples found within the given experiment, round, MCMC tag,
  # emulator tag, and design tag. If `only_valid = TRUE`, drops invalid 
  # runs/chains/iterations - see `get_mcmc_ids()` for details. All MCMC runs
  # within the given set of tags are viewed as random replications of a single
  # experimental setup (e.g., replications stemming from different initial 
  # design samples). All of these are compiled into a single data.table, with 
  # a `rep_id` column added, which is set to the corresponding design_id for
  # each run.

  mcmc_id_dt <- get_mcmc_ids(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE)
  
  mcmc_ids <- unique(mcmc_id_dt$mcmc_id)
  mcmc_tag_dir <- file.path(experiment_dir, paste0("round", round), "mcmc",
                            mcmc_tag)
  
  # Read MCMC samples, and subset chains/itrs based on `mcmc_ids`.
  dt_list <- vector(mode="list", length=length(mcmc_ids))

  for(i in seq_along(dt_list)) {
    id <- mcmc_ids[i]
    design_id <- mcmc_id_dt[mcmc_id==id, design_id][1]
    samp_list <- readRDS(file.path(mcmc_tag_dir, id, "samp.rds"))
    samp_dt <- samp_list$samp
    
    # Subset to include only valid iterations.
    if(only_valid) {
      chain_itr_dt <- mcmc_id_dt[mcmc_id==id, .(chain_idx, itr_start=itr_min, 
                                                itr_stop=itr_max)]
      samp_dt <- select_itr_by_chain(samp_dt, chain_itr_dt)
    }
    
    samp_dt[, rep_id := design_id]
    dt_list[[i]] <- samp_dt
  }
  
  samp_dt_reps <- rbindlist(dt_list, use.names=TRUE)
  return(list(samp=samp_dt_reps, ids=mcmc_id_dt))
}


get_samp_dt_reps_agg <- function(experiment_dir, round, mcmc_tag, em_tag, 
                                 design_tag, only_valid=TRUE, format_long=FALSE,
                                 interval_probs=NULL) {
  # This is similar to `get_samp_dt_reps()`, but here each MCMC run is 
  # aggregated after it is loaded, producing one-dimensional summaries of each
  # variable. Currently, this includes mean, variance, and coverage. If such 
  # summaries are all that is needed, this function is a much more 
  # space-efficient option compared to `get_samp_dt_reps()`.
  #
  # TODO:
  # 1.) add support for using chain weights.
  
  # Define grouping columns for aggregation.
  group_cols <- c("test_label", "param_type", "param_name")
  
  # Determine set of MCMC runs to read.  
  mcmc_id_dt <- get_mcmc_ids(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE)
  
  mcmc_ids <- unique(mcmc_id_dt$mcmc_id)
  mcmc_tag_dir <- file.path(experiment_dir, paste0("round", round), "mcmc",
                            mcmc_tag)
  
  # Separate data.tables will store means/vars and marginal credible intervals.
  stats_list <- vector(mode="list", length=length(mcmc_ids))
  cred_interval_list <- vector(mode="list", length=length(mcmc_ids))
  
  # Read MCMC samples, and subset chains/itrs based on `mcmc_ids`.
  for(i in seq_along(stats_list)) {
    
    # Read samples from run.
    id <- mcmc_ids[i]
    design_id <- mcmc_id_dt[mcmc_id==id, design_id][1]
    samp_list <- readRDS(file.path(mcmc_tag_dir, id, "samp.rds"))
    samp_dt <- samp_list$samp
    
    # Subset to include only valid iterations.
    if(only_valid) {
      chain_itr_dt <- mcmc_id_dt[mcmc_id==id, .(chain_idx, itr_start=itr_min, 
                                                itr_stop=itr_max)]
      samp_dt <- select_itr_by_chain(samp_dt, chain_itr_dt)
    }
    
    # Compute univariate aggregate statistics (mean, variance).
    stat_info <- compute_mcmc_param_stats(samp_dt, subset_samp=FALSE, 
                                          format_long=format_long,
                                          group_cols=group_cols,
                                          interval_probs=interval_probs)
    
    # Means/variances.
    par_stats <- stat_info$par_stats
    par_stats[, rep_id := design_id]
    stats_list[[i]] <- par_stats
    
    # Marginal credible intervals.
    cred_intervals <- stat_info$cred_intervals
    cred_intervals[, rep_id := design_id]
    cred_interval_list[[i]] <- cred_intervals
  }
  
  par_stats_dt <- rbindlist(stats_list, use.names=TRUE)
  cred_intervals_dt <- rbindlist(cred_interval_list, use.names=TRUE)
  
  return(list(par_stats=par_stats_dt, cred_intervals=cred_intervals_dt,
              ids=mcmc_id_dt))
}


get_mcmc_rep_ids <- function(experiment_dir, round, mcmc_tags_prev, 
                             design_tag_prev, em_tag_prev) {
  # Fetches MCMC IDs that are associated with 
  # the given round, MCMC tags (from the previous round), design tag
  # (from the previous round) and emulator tag (from the previous round), all
  # within the specified `experiment_dir`. This is typically intended to 
  # identify "replications" of the same experimental setup. Note that 
  # `mcmc_tags_prev` can be a vector of multiple tags, while the other 
  # arguments should only specify a single tag.
  
  # Previous round - ensure that it exists (i.e., that this is not the first round).
  prev_round <- round - 1L
  assert_that(prev_round > 0)
  
  # Identify proper directories.
  prev_round_dir <- file.path(experiment_dir, paste0("round", prev_round))
  mcmc_dir_prev <- file.path(prev_round_dir, "mcmc")
  em_dir_prev <- file.path(prev_round_dir, "em")
  
  # Restrict to specified em tag and design tag.
  em_id_map <- fread(file.path(em_dir_prev, em_tag_prev, "id_map.csv"))
  em_id_map <- em_id_map[design_tag==design_tag_prev]
  id_map <- NULL
  
  for(mcmc_tag in mcmc_tags_prev) {
    mcmc_id_map <- fread(file.path(mcmc_dir_prev, mcmc_tag, "id_map.csv"))
    mcmc_id_map <- mcmc_id_map[em_tag==em_tag_prev]
    mcmc_id_map <- data.table::merge.data.table(mcmc_id_map, em_id_map, by="em_id",
                                                all=FALSE)
    mcmc_id_map[, mcmc_tag := mcmc_tag]
    
    if(is.null(id_map)) id_map <- copy(mcmc_id_map)
    else id_map <- rbindlist(list(id_map, mcmc_id_map), use.names=TRUE)
  }
  
  return(id_map)
}


# ------------------------------------------------------------------------------
# Loading/Assembling sequential design/acquisition results.
# ------------------------------------------------------------------------------

load_acq_data <- function(experiment_dir, round, acq_id, mcmc_tag_prev,
                          mcmc_id_prev) {
  # Loads a single acquisition results file.
  
  results_path <- file.path(experiment_dir, paste0("round", round), 
                            "design", paste0("acq_", acq_id), mcmc_tag_prev,
                            mcmc_id_prev, "acq_results.rds")
  acq_results <- readRDS(results_path)
  return(acq_results)
}


process_acq_results <- function(acq_results) {
  # Given a single acquisition results file, converts all of the tracked 
  # quantities into a data.table, and also returns a separate data.table with 
  # the responses at the acquired points and the values of the acquisition 
  # function at each iteration. This latter data.table will have info for
  # every iteration, while the former may not, depending on the interval at 
  # which the tracked quantities were computed.
  
  tracked_quantities <- extract_acq_tracked_quantities_table(acq_results)
  itr_info <- extract_acq_itr_info(acq_results)
  
  return(list(tracked_quantities=tracked_quantities, itr_info=itr_info))
}


extract_acq_tracked_quantities_table <- function(acq_results) {
  # Returns data.table with columns: itr, name, metric, val. In the current
  # setting "name" corresponds to the validation dataset used (prior vs.
  # posterior validation). This function assumes a specific structure
  # as I have currently set it up, but should be generalized in the future.
  
  dt <- data.table(itr=integer(), name=character(), metric=character(),
                   val=numeric())
  tracked_quantities <- acq_results$tracking_list$computed_quantities
  
  # Tracked quantities may not be computed every iteration.
  itrs_tracked_str <- names(tracked_quantities)
  itrs_tracked <- sapply(strsplit(itrs_tracked_str, "_", fixed=TRUE), 
                         function(x) as.integer(x[2]))
  
  # Aggregated metrics.
  log_scores_post <- sapply(itrs_tracked_str, 
                            function(itr) drop(tracked_quantities[[itr]]$agg_post))
  log_scores_prior <- sapply(itrs_tracked_str, 
                             function(itr) drop(tracked_quantities[[itr]]$agg_prior))
  dt_agg_post <- data.table(itr=itrs_tracked, name="post", metric="log_score",
                            val=log_scores_post)
  dt_agg_prior <- data.table(itr=itrs_tracked, name="prior", metric="log_score",
                             val=log_scores_post)
  dt <- rbindlist(list(dt, dt_agg_post, dt_agg_prior), use.names=TRUE)
  
  # Pointwise metrics (which have already been aggregated).
  group_names <- c("pw_prior", "pw_post")
  
  for(i in seq_along(tracked_quantities)) {
    itr <- itrs_tracked[i]
    itr_quantities <- tracked_quantities[[i]]
    
    for(nm in group_names) {
      dt_curr <- copy(itr_quantities[[nm]])
      setnames(dt_curr, c("func", "mean"), c("metric", "val"))
      nm_short <- ifelse(nm=="pw_post", "post", "prior")
      dt_curr[, `:=`(name=nm_short, itr=itr)]
      dt_curr[metric=="mse", `:=`(metric="rmse", val=sqrt(val))]
      dt <- rbindlist(list(dt, dt_curr), use.names=TRUE)
    }
  }
  
  return(dt)  
}


extract_acq_itr_info <- function(acq_results) {
  # Stores data.table with columns: itr, response, acq_val.
  
  data.table(itr = seq_along(acq_results$responses),
             response = acq_results$responses,
             acq_val = acq_results$tracking_list$acq_val)
}


process_acq_data_reps <- function(experiment_dir, round, acq_ids, mcmc_tags_prev,
                                  design_tag_prev, em_tag_prev) {
  # Loads acquisition results for all MCMC IDs that are associated with 
  # the given round, acq IDs, MCMC tags (from the previous round), design tag
  # (from the previous round) and emulator tag (from the previous round), all
  # within the specified `experiment_dir`. This is typically intended to 
  # identify "replications" of the same experimental setup. Note that 
  # `mcmc_tags_prev` and `acq_ids` can be vectors with multiple elements, 
  # while the other arguments should only specify one tag/ID each.
  #
  # Returns the same two data.tables as `process_acq_results()`, where the
  # information from all of the loaded runs have been stacked together. 
  # The following columns are also added: acq_id, mcmc_tag, mcmc_id.
  # Also returns the ID map produced by `get_mcmc_rep_ids()`.
  
  dt_track <- dt_itr <- NULL
  
  # Fetch the MCMC IDs.
  id_map <- get_mcmc_rep_ids(experiment_dir, round, mcmc_tags_prev, 
                             design_tag_prev, em_tag_prev)
  mcmc_tag_id <- unique(id_map[, .(mcmc_tag, mcmc_id)])
  
  for(acq_id in acq_ids) {
    for(i in 1:nrow(mcmc_tag_id)) {
      mcmc_tag_prev <- mcmc_tag_id[i,mcmc_tag]
      mcmc_id_prev <- mcmc_tag_id[i,mcmc_id]
      
      # Process results from the acquisition run.
      acq_results <- load_acq_data(experiment_dir, round, acq_id, 
                                   mcmc_tag_prev, mcmc_id_prev)
      results_list <- process_acq_results(acq_results)
      
      # Append to tracked quantities table.
      tracked_quantities <- results_list$tracked_quantities
      tracked_quantities[, `:=`(acq_id=acq_id, mcmc_tag=mcmc_tag_prev,
                                mcmc_id=mcmc_id_prev)]
      if(is.null(dt_track)) dt_track <- copy(tracked_quantities)
      else dt_track <- rbindlist(list(dt_track, tracked_quantities))
      
      # Append to iteration info table.
      itr_info <- results_list$itr_info
      itr_info[, `:=`(acq_id=acq_id, mcmc_tag=mcmc_tag_prev, mcmc_id=mcmc_id_prev)]
      if(is.null(dt_itr)) dt_itr <- copy(itr_info)
      else dt_itr <- rbindlist(list(dt_itr, itr_info))
    }
  }
  
  return(list(dt_track=dt_track, dt_itr=dt_itr, id_map=id_map,
              design_tag_prev=design_tag_prev, em_tag_prev=em_tag_prev))
}


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------








