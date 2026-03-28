#!/bin/bash -l
#$ -N pde_validate_ep
#$ -j y
#$ -l h_rt=24:00:00
#$ -l mem_per_core=16G
#$ -P bayesij
#$ -pe omp 1
#
# Validate EP baseline for specific PDE replicates.
# Runs MCwMH convergence, RFF convergence, and chain quality studies.
#
# Usage:
#   cd experiments/elliptic_pde
#   qsub submit_validate.sh
#
# Edit the EXPERIMENT_NAME, N_DESIGN, and REP variables below before submitting.

# --- USER SETTINGS ---
EXPERIMENT_NAME="pde_local_test2"
N_DESIGN=4
REP=0
# ----------------------

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"

echo "JOB_ID=$JOB_ID"
hostname
python --version

# Scratch space for XLA
export TMPDIR=/scratch/$USER/$JOB_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

# Single-threaded computation
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TF_CPP_MIN_LOG_LEVEL=2

# Per-job JAX cache
export JAX_CACHE_DIR=/scratch/$USER/jax_cache_$JOB_ID
mkdir -p $JAX_CACHE_DIR
trap "rm -rf $TMPDIR $JAX_CACHE_DIR" EXIT

OUTPUT_DIR="${REPO_DIR}/out/${EXPERIMENT_NAME}/validation"

cd "${SGE_O_WORKDIR}"

exec python -u validate_ep.py \
    --experiment-name ${EXPERIMENT_NAME} \
    --n-design ${N_DESIGN} \
    --rep ${REP} \
    --output-dir ${OUTPUT_DIR} \
    --heavy-n-chains 500 \
    --heavy-n-samp 200 \
    --num-rff-values 500 1000 2000
