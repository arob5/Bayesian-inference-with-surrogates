#
# inv_prob_setup.r
#
# Defines a Bayesian inverse problem based around estimating the parameters
# of the Very Simple Ecosystem Model (VSEM), a toy carbon cycle model 
# consisting of a system of ODEs.
# This script is intended to be called by `run_inv_prob_setup.r`.
# Note that the name of this script is formatted as `<experiment_tag>_setup.r`,
# as required by the `run_inv_prob_setup.r` script.
#

get_inv_prob <- function() {
  
  # ------------------------------------------------------------------------------
  # Inverse problem setup 
  # ------------------------------------------------------------------------------
  
  # Helper function sets up the inverse problem.
  inv_prob <- get_vsem_test_paper(default_conditional=FALSE, 
                                  default_normalize=TRUE)
  
  return(inv_prob)
}

