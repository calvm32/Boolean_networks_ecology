#!/bin/bash
#SBATCH --job-name=phase_diagram_2var
#SBATCH --output=results_and_data/20261101_210108_phase_diagram_2var_outbreakseverity:alpha_vs_Tinf_LAT1NMW/logs/job.out
#SBATCH --error=results_and_data/20261101_210108_phase_diagram_2var_outbreakseverity:alpha_vs_Tinf_LAT1NMW/logs/job.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH --time=2:00:00
#SBATCH --mem=80G

# Setup compute node environment
module purge
# MINIMAL CHANGE: Ensured openmpi is loaded on the compute node
module load compiler/gcc/11 openmpi/4.1 python/3.10

# Activate virtual environment
source "/home/velcsov/cheldt/envs/bn_ecology_env/bin/activate"

# Add top-level project directory to Python search path
export PYTHONPATH="/home/velcsov/cheldt/BN_ecology${PYTHONPATH:+:${PYTHONPATH}}"

# Export variables for Python multiprocessing (keeping variables for backwards compatibility)
export SIM_OUTPUT_DIR="results_and_data/20261101_210108_phase_diagram_2var_outbreakseverity:alpha_vs_Tinf_LAT1NMW"
export SIM_NUM_CORES="20"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

echo "==================================================="
echo "Starting Execution: phase_diagram_2var"
echo "==================================================="

mpirun python3 "/home/velcsov/cheldt/BN_ecology/simulate/simulate_CURRENT/compare_regimes/phase_diagram_2var.py"
