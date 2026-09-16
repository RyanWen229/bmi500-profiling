#!/usr/bin/env bash
# Execute one dataset run, retaining both application timing and OS resource use.
set -euo pipefail

dataset=${1:?usage: run_pbmc_experiment.sh '{pbmc3k|pbmc6k|pbmc10k} [threads]'}
threads=${2:-1}
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
log_dir="${root_dir}/logs"
mkdir -p "${log_dir}" "${root_dir}/results" "${root_dir}/profiles"

source /opt/scratchspace/shenghan/miniconda3_clean/etc/profile.d/conda.sh
conda activate bmi500
export OMP_NUM_THREADS="${threads}"
export OPENBLAS_NUM_THREADS="${threads}"
export MKL_NUM_THREADS="${threads}"
export NUMBA_NUM_THREADS="${threads}"

log_file="${log_dir}/${dataset}.console.log"
{
  echo "===== $(date --iso-8601=seconds) ${dataset} ====="
  echo "host=$(hostname) threads=${threads}"
  /usr/bin/time -v python "${root_dir}/scanpy_pbmc.py" \
    --data-dir "${root_dir}/data" --data-set "${dataset}" \
    --out-dir "${root_dir}/results" --profile-dir "${root_dir}/profiles" \
    --num-threads "${threads}"
} 2>&1 | tee "${log_file}"
