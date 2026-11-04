#!/bin/bash
#SBATCH --job-name=sobol_analysis
#SBATCH --output=results_and_data/20261012_192320_sobol_analysis_lat1_N2_18/logs/job.out
#SBATCH --error=results_and_data/20261012_192320_sobol_analysis_lat1_N2_18/logs/job.err
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=8
#SBATCH --time=20:00:00
#SBATCH --mem=32G

# Setup compute node environment
module purge
# MINIMAL CHANGE: Ensured openmpi is loaded on the compute node
module load compiler/gcc/11 openmpi/4.1 python/3.10

# Activate virtual environment
source "/home/velcsov/cheldt/envs/bn_ecology_env/bin/activate"

# Add top-level project directory to Python search path
export PYTHONPATH="/home/velcsov/cheldt/BN_ecology${PYTHONPATH:+:${PYTHONPATH}}"

# Export variables for Python multiprocessing (keeping variables for backwards compatibility)
export SIM_OUTPUT_DIR="results_and_data/20261012_192320_sobol_analysis_lat1_N2_18"
export SIM_NUM_CORES="8"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

echo "==================================================="
echo "Starting Execution: sobol_analysis"
echo "==================================================="

mpirun python3 "/home/velcsov/cheldt/BN_ecology/simulate/simulate_CURRENT/compare_regimes/sobol_analysis.py"
