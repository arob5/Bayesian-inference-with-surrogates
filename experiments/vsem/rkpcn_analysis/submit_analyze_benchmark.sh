#!/bin/bash -l
#$ -N rkpcn_analyze
#$ -j y
#$ -l h_rt=2:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 1
#
# Post-process RKPCN benchmark results: summary table, W2, plots.
#
# Usage:
#   cd experiments/vsem
#   qsub rkpcn_analysis/submit_analyze_benchmark.sh
#
# Override defaults via qsub -v:
#   qsub -v SETUP=gp_N8,REP=3 rkpcn_analysis/submit_analyze_benchmark.sh
#
# Configurable via environment variables:
#   EXPERIMENT  - experiment name (default: vsem)
#   SETUP       - setup name (default: clip_gp_N4)
#   REP         - replicate index (default: 0)
#   OUTPUT_DIR  - benchmark output dir (default: auto from setup/rep)
#   NO_W2       - set to "1" to skip W2 computation (default: 0)

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"

# Thread-limiting for JAX (W2 via Sinkhorn can be multi-threaded)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"

# Defaults
EXPERIMENT="${EXPERIMENT:-vsem}"
SETUP="${SETUP:-clip_gp_N4}"
REP="${REP:-0}"
NO_W2="${NO_W2:-0}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${REPO_DIR}/out/${EXPERIMENT}/benchmarks/${SETUP}_rep${REP}"
fi

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "Analyzing: $OUTPUT_DIR"
echo "Experiment data: $EXPERIMENT / $SETUP / rep$REP"

cd "${SGE_O_WORKDIR}"

W2_FLAG=""
if [ "$NO_W2" = "1" ]; then
    W2_FLAG="--no-w2"
fi

exec python -u -m rkpcn_analysis.benchmark analyze \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT" \
    --setup "$SETUP" \
    --rep "$REP" \
    $W2_FLAG
