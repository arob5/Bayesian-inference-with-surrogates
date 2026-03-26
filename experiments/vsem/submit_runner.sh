#!/bin/bash -l
#$ -N vsem_experiment
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 1
#$ -t 1-60 -tc 1    # 6 setups x 10 chunks = 60 tasks; cap concurrency
#
# Array job for executing VSEM experiment. It is the user's responsibility
# for setting the task IDs (e.g., `-t 1-60`) and the batch size
# (e.g., `--rep-chunk-size 10`) in a compatible manner.
#
# With 6 setups ({gp,clip_gp} x {N=4,N=8,N=16}) and 100 reps per setup,
# a chunk size of 10 yields 60 tasks (task IDs 1-60).
#
# Issues with JAX running on CPU clusters:
# https://github.com/jax-ml/jax/issues/1539
# https://github.com/jax-ml/jax/issues/16215

# Activate virtual environment - update this path for your environment
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"

# Debug and version info
echo "JOB_ID=$JOB_ID SGE_TASK_ID=$SGE_TASK_ID"
hostname
python --version
echo "Git commit: $(git rev-parse HEAD)"

# Scratch space for XLA
export TMPDIR=/scratch/$USER/$JOB_ID.$SGE_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

# Force CPU-only JAX
export JAX_PLATFORMS=cpu

# Encourage single-threaded computation to avoid XLA compilation conflicts
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"
export TF_CPP_MIN_LOG_LEVEL=2

# Per-job JAX cache to avoid compilation conflicts between concurrent jobs
export JAX_CACHE_DIR=/scratch/$USER/jax_cache_$JOB_ID.$SGE_TASK_ID
mkdir -p $JAX_CACHE_DIR
trap "rm -rf $TMPDIR $JAX_CACHE_DIR" EXIT

# Convert SGE 1-based task ID to 0-based Python indexing
TASK_ID=$((SGE_TASK_ID - 1))

# Sleep 0-30 seconds to stagger JAX compilation across concurrent jobs
sleep $((RANDOM % 30))

exec python -u "${REPO_DIR}/experiments/vsem/runner.py" \
    --task-id ${TASK_ID} \
    --rep-chunk-size 10 \
    --experiment-name vsem
