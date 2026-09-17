#!/bin/bash
# Automated Batch Execution Tool for All Sites (Slurm Job Array)

set -e

PROJECT_DIR=$(pwd)
DATA_FILE="simulate/data.py"
OUTPUT_BASE="results"
VENV_DIR="$HOME/envs/bn_ecology_env"
SCRIPT_PATH="simulate/simulate_CURRENT/running_solvers/fit_data_parallel_ML.py"

# Extract all site function names from simulate/data.py
SITE_LIST=($(grep -E '^def Site_' "$DATA_FILE" | sed -E 's/def (Site_[A-Za-z0-9_]+).*/\1/'))
NUM_SITES=${#SITE_LIST[@]}

if [ "$NUM_SITES" -eq 0 ]; then
    echo "Error: No Site_* functions found in $DATA_FILE"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_DIR="${OUTPUT_BASE}/batch_${TIMESTAMP}"
mkdir -p "${BATCH_DIR}/logs" "${BATCH_DIR}/data"

# Save site list mapping for the Slurm Array
SITES_FILE="${BATCH_DIR}/sites.txt"
printf "%s\n" "${SITE_LIST[@]}" > "$SITES_FILE"

echo "==================================================="
echo "Found ${NUM_SITES} sites in ${DATA_FILE}"
echo "Batch Output: ${BATCH_DIR}"
echo "==================================================="

# Generate Slurm Array Batch File
SLURM_SCRIPT="${BATCH_DIR}/slurm_array.sh"

cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=bat_pso
#SBATCH --output=${BATCH_DIR}/logs/site_%A_%a.out
#SBATCH --error=${BATCH_DIR}/logs/site_%A_%a.err
#SBATCH --array=1-${NUM_SITES}%10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=20:00:00
#SBATCH --mem=16G

# Determine site name for this array task
SITE_NAME=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${SITES_FILE}")

module purge
module load compiler/gcc/11 openmpi/4.1 python/3.10

source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${PROJECT_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}"
export SITE_NAME="\${SITE_NAME}"
export SIM_OUTPUT_DIR="${BATCH_DIR}/data/\${SITE_NAME}"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

mkdir -p "\${SIM_OUTPUT_DIR}"

echo "==================================================="
echo "Starting Task \${SLURM_ARRAY_TASK_ID}/${NUM_SITES}: \${SITE_NAME}"
echo "Running on host: \$(hostname)"
echo "==================================================="

mpirun -np 16 python3 "${PROJECT_DIR}/${SCRIPT_PATH}"
EOF

# Submit to Slurm
echo "Submitting Slurm Job Array..."
sbatch "$SLURM_SCRIPT"
echo "Job array submitted. Monitor with: squeue -u \$(whoami)"