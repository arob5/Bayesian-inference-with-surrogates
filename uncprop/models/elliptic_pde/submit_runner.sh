#!/bin/bash
#$ -N pde_experiment
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=4G
#$ -P dietzelab 
#$ -l buyin
#$ -pe omp 1
#$ -t 1-30
#
# Array job for executing PDE experiment. It is the users responsibility
# for setting the task IDs (e.g., `-t 1-30`) and the batch size 
# (e.g., `--rep-chunk-size 10`) in a compatible manner.
#
# Example:
# For 3 different design settings, with 100 reps per setting. Batch size of
# 10 implies 30 jobs, so the task IDs should be specified as `-t 1-30`.

source /projectnb/dietzelab/arober/Bayesian-inference-with-surrogates/.venv/bin/activate

# convert to 0-based Python indexing
TASK_ID=$((SGE_TASK_ID - 1))

python runner.py \
    --task-id ${TASK_ID} \
    --rep-chunk-size 10
