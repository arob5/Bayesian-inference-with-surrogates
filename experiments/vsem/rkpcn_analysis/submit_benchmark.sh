#!/bin/bash -l
#$ -N rkpcn_bench
#$ -j y
#$ -l h_rt=6:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 1
#
# Submit an RKPCN benchmark run on a VSEM replicate.
#
# Usage:
#   cd experiments/vsem
#   qsub rkpcn_analysis/submit_benchmark.sh
#
# Override defaults via qsub -v:
#   qsub -v SETUP=gp_N8,REP=3,CONFIG=my_variants.yaml rkpcn_analysis/submit_benchmark.sh
#
# Configurable via environment variables:
#   EXPERIMENT  - experiment name (default: vsem)
#   SETUP       - setup name (default: clip_gp_N4)
#   REP         - replicate index (default: 0)
#   CONFIG      - YAML config file (default: rkpcn_analysis/example_benchmark.yaml)
#   OUTPUT_DIR  - output directory (default: auto-generated from setup/rep)
#   SEED        - PRNG seed (default: 42)
#   SAVE_TRACE  - set to "1" to save full traces (default: 0)

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"

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
CONFIG="${CONFIG:-rkpcn_analysis/example_benchmark.yaml}"
SEED="${SEED:-42}"
SAVE_TRACE="${SAVE_TRACE:-0}"

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${REPO_DIR}/out/${EXPERIMENT}/benchmarks/${SETUP}_rep${REP}"
fi

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "EXPERIMENT=$EXPERIMENT SETUP=$SETUP REP=$REP"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "${SGE_O_WORKDIR}"

TRACE_FLAG=""
if [ "$SAVE_TRACE" = "1" ]; then
    TRACE_FLAG="--save-trace"
fi

exec python -u -m rkpcn_analysis.benchmark run \
    --experiment-name "$EXPERIMENT" \
    --setup "$SETUP" \
    --rep "$REP" \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $TRACE_FLAG
