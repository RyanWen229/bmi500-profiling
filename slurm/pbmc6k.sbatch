#!/usr/bin/env bash
#SBATCH --job-name=pbmc6k-profile
#SBATCH --output=slurm-pbmc6k-%j.out
#SBATCH --error=slurm-pbmc6k-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_pbmc_experiment.sh pbmc6k "${SLURM_CPUS_PER_TASK}"
