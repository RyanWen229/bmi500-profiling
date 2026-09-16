#!/usr/bin/env bash
#SBATCH --job-name=pbmc10k-profile
#SBATCH --output=slurm-pbmc10k-%j.out
#SBATCH --error=slurm-pbmc10k-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_pbmc_experiment.sh pbmc10k "${SLURM_CPUS_PER_TASK}"
