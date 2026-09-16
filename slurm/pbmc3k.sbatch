#!/usr/bin/env bash
#SBATCH --job-name=pbmc3k-profile
#SBATCH --output=slurm-pbmc3k-%j.out
#SBATCH --error=slurm-pbmc3k-%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
bash scripts/run_pbmc_experiment.sh pbmc3k "${SLURM_CPUS_PER_TASK}"
