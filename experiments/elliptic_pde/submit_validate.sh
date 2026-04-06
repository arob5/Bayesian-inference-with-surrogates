#!/bin/bash -l
#$ -N pde_validate_ep
#$ -j y
#$ -l h_rt=24:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 2
#
# Validate EP baseline for specific PDE replicates.
#
# Usage:
#   cd experiments/elliptic_pde
#
#   # Run just chain quality (fast, ~5 min):
#   qsub -v STUDIES=chain_quality submit_validate.sh
#
#   # Run all 3 studies (slow, several hours):
#   qsub submit_validate.sh
#
#   # Override experiment/design/rep:
#   qsub -v EXPERIMENT_NAME=pde_experiment,N_DESIGN=20,REP=5 submit_validate.sh
#
# All variables below can be overridden via qsub -v.

# --- USER SETTINGS (override with qsub -v VAR=value) ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pde_local_test2}"
N_DESIGN="${N_DESIGN:-4}"
REP="${REP:-0}"
STUDIES="${STUDIES:-all}"
# ----------------------

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

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

# Build studies argument
if [ "$STUDIES" = "all" ]; then
    STUDIES_ARG=""
else
    STUDIES_ARG="--studies ${STUDIES}"
fi

exec python -u validate_ep.py \
    --experiment-name ${EXPERIMENT_NAME} \
    --n-design ${N_DESIGN} \
    --rep ${REP} \
    --output-dir ${OUTPUT_DIR} \
    --heavy-n-chains 500 \
    --heavy-n-samp 200 \
    --num-rff-values 500 1000 2000 \
    ${STUDIES_ARG}
