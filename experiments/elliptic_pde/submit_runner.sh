#!/bin/bash
#$ -N pde_experiment
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=12G
#$ -P robustvb
#$ -pe omp 1
#$ -t 1-30 -tc 1    # 3 design sizes x 10 chunks = 30 tasks; cap concurrency
#
# Array job for executing PDE experiment. The task IDs and batch size
# must be compatible: 3 design sizes × ceil(num_reps / rep_chunk_size)
# tasks total. With 100 reps and chunk size 10: 3 × 10 = 30 tasks.
#
# RKPCN samplers (rho=0.0, 0.9, 0.95, 0.99) are included via run_kwargs.
#
# Issues with JAX running on CPU clusters:
# https://github.com/jax-ml/jax/issues/1539
# https://github.com/jax-ml/jax/issues/16215

# SGE copies scripts to a spool directory before execution, so dirname "$0"
# does not point to the repo. Use SGE_O_WORKDIR (the submission directory)
# instead. This assumes qsub is run from experiments/<experiment_name>/.
REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

# Debug and version info
echo "JOB_ID=$JOB_ID SGE_TASK_ID=$SGE_TASK_ID"
hostname
python --version
echo "Git commit: $(git rev-parse HEAD)"

# Scratch space for XLA
export TMPDIR=/scratch/$USER/$JOB_ID.$SGE_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

# encourage single-threaded computation
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TF_CPP_MIN_LOG_LEVEL=2

# per job JAX cache
export JAX_CACHE_DIR=/scratch/$USER/jax_cache_$JOB_ID.$SGE_TASK_ID
mkdir -p $JAX_CACHE_DIR
trap "rm -rf $TMPDIR $JAX_CACHE_DIR" EXIT

# convert to 0-based Python indexing
TASK_ID=$((SGE_TASK_ID - 1))

# sleep between 0-30 seconds - trying to avoid JAX compilation issues due to concurrency
sleep $((RANDOM % 30))

exec python -u "${REPO_DIR}/experiments/elliptic_pde/runner.py" \
    --task-id ${TASK_ID} \
    --rep-chunk-size 10 \
    --experiment-name pde_experiment