#!/bin/bash -l
#$ -N rkpcn_smoke
#$ -j y
#$ -l h_rt=0:30:00
#$ -l mem_per_core=8G
#$ -P robustvb
#$ -pe omp 1
#
# Smoke test for the multi-chain RKPCN + EP-init pipeline.
# Runs a single variant (4 chains, rho=0.9, adaptive, ep_direct_sampling
# init, Pritchard mode weights) with small n_total/n_burnin — just to
# verify the new code paths work end-to-end before launching the full
# benchmark sweep.
#
# Usage:
#   cd experiments/vsem
#   qsub rkpcn_analysis/submit_smoke_test.sh
#
# Override defaults via qsub -v:
#   qsub -v SETUP=gp_N8,REP=3 rkpcn_analysis/submit_smoke_test.sh

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

# Thread-limiting for JAX
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TMPDIR=/scratch/$USER/$JOB_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

# Defaults (override via qsub -v)
EXPERIMENT="${EXPERIMENT:-vsem}"
SETUP="${SETUP:-clip_gp_N4}"
REP="${REP:-0}"
CONFIG="${CONFIG:-rkpcn_analysis/smoke_test.yaml}"
SEED="${SEED:-42}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${REPO_DIR}/out/${EXPERIMENT}/benchmarks/smoke_test_${SETUP}_rep${REP}"
fi

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "EXPERIMENT=$EXPERIMENT SETUP=$SETUP REP=$REP"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "${SGE_O_WORKDIR}"

exec python -u -m rkpcn_analysis.benchmark run \
    --experiment-name "$EXPERIMENT" \
    --setup "$SETUP" \
    --rep "$REP" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --save-trace
