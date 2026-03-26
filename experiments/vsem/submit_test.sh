#!/bin/bash -l
#$ -N vsem_test
#$ -j y
#$ -l h_rt=2:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 1
#$ -t 1-2 -tc 1
#
# Small test run: 2 tasks covering reps 0-4 of gp_N4 (setup_idx=0).
# Task 1 → reps [0,1,2,3,4], Task 2 → reps [5,6,7,8,9]
# (with rep_chunk_size=5, only the first setup is covered by task IDs 1-2)
#
# After this completes, check output at:
#   out/vsem_scc_test/gp_N4/rep{0..9}/samples.npz
#
# To verify:
#   cd experiments/vsem
#   python analyze.py --experiment-name vsem_scc_test --num-reps 100
#
# Note: uses experiment-name "vsem_scc_test" to avoid interfering with
# the real experiment output.

# SGE copies scripts to a spool directory before execution, so dirname "$0"
# does not point to the repo. Use SGE_O_WORKDIR (the submission directory)
# instead. This assumes qsub is run from experiments/vsem/.
REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"

echo "JOB_ID=$JOB_ID SGE_TASK_ID=$SGE_TASK_ID"
hostname
python --version
echo "Git commit: $(git -C ${REPO_DIR} rev-parse HEAD)"

# Scratch space for XLA
export TMPDIR=/scratch/$USER/$JOB_ID.$SGE_TASK_ID
mkdir -p $TMPDIR

# Force CPU-only JAX
export JAX_PLATFORMS=cpu

# Single-threaded computation
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1 --xla_force_host_platform_device_count=1"
export TF_CPP_MIN_LOG_LEVEL=2

# Per-job JAX cache
export JAX_CACHE_DIR=/scratch/$USER/jax_cache_$JOB_ID.$SGE_TASK_ID
mkdir -p $JAX_CACHE_DIR
trap "rm -rf $TMPDIR $JAX_CACHE_DIR" EXIT

TASK_ID=$((SGE_TASK_ID - 1))

sleep $((RANDOM % 10))

exec python -u "${REPO_DIR}/experiments/vsem/runner.py" \
    --task-id ${TASK_ID} \
    --rep-chunk-size 5 \
    --experiment-name vsem_scc_test
