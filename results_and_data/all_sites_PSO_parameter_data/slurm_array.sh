#!/bin/bash
#SBATCH --job-name=bat_pso
#SBATCH --output=results_and_data/batch_20260922_180909/logs/site_%A_%a.out
#SBATCH --error=results_and_data/batch_20260922_180909/logs/site_%A_%a.err
#SBATCH --array=1-35%10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32 
#SBATCH --time=10:00:00
#SBATCH --mem=16G

# Determine site name for this array task
SITE_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "results_and_data/batch_20260922_180909/sites.txt")

module purge
module load compiler/gcc/11 openmpi/4.1 python/3.10

source "/home/velcsov/cheldt/envs/bn_ecology_env/bin/activate"

export PYTHONPATH="/home/velcsov/cheldt/BN_ecology${PYTHONPATH:+:${PYTHONPATH}}"
export SITE_NAME="${SITE_NAME}"
export SIM_OUTPUT_DIR="results_and_data/batch_20260922_180909/data/${SITE_NAME}"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

mkdir -p "${SIM_OUTPUT_DIR}"

echo "==================================================="
echo "Starting Task ${SLURM_ARRAY_TASK_ID}/35: ${SITE_NAME}"
echo "Running on host: $(hostname)"
echo "==================================================="

# Update the -np flag to match --ntasks-per-node above
mpirun -np 32 python3 "/home/velcsov/cheldt/BN_ecology/simulate/simulate_CURRENT/running_solvers/fit_data_parallel_ML_SLURM.py"
