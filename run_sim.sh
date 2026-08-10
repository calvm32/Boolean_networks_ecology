#!/bin/bash
# BN_Ecology Simulation Execution Tool

set -e

# Configuration & Paths
PROJECT_DIR=$(pwd)
OUTPUT_BASE="results"
SIM_DIR="simulate/simulate_CURRENT"

module load python/3.10
python3 -m venv ~/envs/bn_ecology_env
source ~/envs/bn_ecology_env/bin/activate
pip install -r requirements.txt

# Styling & Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "BN_Ecology Simulation Execution Tool"
echo "==================================================="
echo -e "${NC}"

# Fix the /dev/null typo from the original script
MAX_CORES=$(nproc 2> /dev/null || echo 64)

# Prevent the Exit Code 255 error by verifying the image exists early
if [ ! -f "$IMAGE" ]; then
    echo -e "${RED}${BOLD}✗ Error: Container image not found!${NC}"
    echo -e "Looking for: ${BOLD}$IMAGE${NC}\n"
    echo -e "Please ensure the .sif file is in your current directory, or define it:"
    echo -e "  ${YELLOW}export IMAGE=/path/to/your/container.sif${NC}\n"
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Script Selection

mapfile -t SCRIPTS < <(find "$SIM_DIR" -type f -name "*.py" ! -name "__init__.py" ! -name "helper_funcs.py" ! -name "rules*.py" ! -name "working_params.py" | sort)

if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No python scripts found in $SIM_DIR${NC}"
    exit 1
fi

echo -e "${YELLOW}Select a script to run:${NC}"
PS3=$'\n\033[1;36mEnter option number: \033[0m'

select SCRIPT_PATH in "${SCRIPTS[@]}" "Quit"; do
    if [[ "$SCRIPT_PATH" == "Quit" ]]; then
        echo "Exiting..."
        exit 0
    elif [[ -n "$SCRIPT_PATH" ]]; then
        SCRIPT_NAME=$(basename "$SCRIPT_PATH" .py)
        break
    else
        echo -e "${RED}Invalid selection. Try again.${NC}"
    fi
done

# Resource Allocation

echo -e "\n${MAGENTA}${BOLD}⚡ Resource Configuration${NC}"

# Request Cores
read -p "$(echo -e "${CYAN}  Enter number of cores to allocate (1-$MAX_CORES) [Default: 1]: ${NC}")" USER_CORES
NUM_CORES=${USER_CORES:-1}
if ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]] || [ "$NUM_CORES" -lt 1 ]; then
    NUM_CORES=1
fi

# Request Walltime (Crucial for SLURM queuing)
read -p "$(echo -e "${CYAN}  Enter maximum walltime (HH:MM:SS) [Default: 12:00:00]: ${NC}")" USER_TIME
WALLTIME=${USER_TIME:-12:00:00}

# Calculate Memory (Standard 4GB per core rule of thumb; adjust as needed)
MEM_GB=$((NUM_CORES * 4))

# Organization & Environment

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"
mkdir -p "$RUN_OUT_DIR/data" "$RUN_OUT_DIR/figures" "$RUN_OUT_DIR/logs"

BATCH_FILE="$RUN_OUT_DIR/submit_${SCRIPT_NAME}.sh"

# Generate SLURM Batch Script
# We generate this file dynamically so a permanent record of the exact 
# run conditions is saved alongside the data for publication reproducibility.

cat <<EOF > "$BATCH_FILE"
#!/bin/bash
#SBATCH --job-name=${SCRIPT_NAME}
#SBATCH --output=${RUN_OUT_DIR}/logs/job.out
#SBATCH --error=${RUN_OUT_DIR}/logs/job.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${NUM_CORES}
#SBATCH --time=${WALLTIME}
#SBATCH --mem=${MEM_GB}G

module purge
module load apptainer compiler/gcc/11 openmpi/4.1

set -e

# Pass parameters into the Apptainer environment
export SIM_OUTPUT_DIR="${RUN_OUT_DIR}"
export SIM_NUM_CORES="${NUM_CORES}"
export OMP_NUM_THREADS="${NUM_CORES}"
export MKL_NUM_THREADS="${NUM_CORES}"
export OPENBLAS_NUM_THREADS="${NUM_CORES}"

echo "==================================================="
echo "Starting Execution: ${SCRIPT_NAME}"
echo "==================================================="

apptainer exec \\
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \\
    --pwd "${PROJECT_DIR}" \\
    "${IMAGE}" \\
    python3 "${SCRIPT_PATH}"
EOF

# Generate JSON Run Record
cat <<EOF > "$RUN_OUT_DIR/run_info.json"
{
  "script": "$SCRIPT_PATH",
  "timestamp": "$TIMESTAMP",
  "allocated_cores": $NUM_CORES,
  "walltime": "$WALLTIME",
  "memory_gb": $MEM_GB,
  "container_image": "$IMAGE",
  "user": "$(whoami)",
  "hostname": "$(hostname)"
}
EOF

# Queue Submission

echo -e "\n${GREEN}✓ Execution Environment Prepared${NC}"
echo -e "==================================================="
echo -e "   ${BOLD}Script:${NC}    $SCRIPT_PATH"
echo -e "   ${BOLD}Resources:${NC} $NUM_CORES core(s), $MEM_GB GB RAM, $WALLTIME limit"
echo -e "   ${BOLD}Output:${NC}    $RUN_OUT_DIR"
echo -e "==================================================="

if [ -z "$SLURM_JOB_ID" ]; then
    echo -e "${YELLOW}Submitting job to Slurm queue...${NC}"
    sbatch "$BATCH_FILE"
    echo -e "${CYAN}Use 'squeue -u \$(whoami)' to check job status.${NC}\n"
else
    # If already running inside an interactive salloc/srun node
    echo -e "${YELLOW}Running directly in current Slurm allocation...${NC}"
    bash "$BATCH_FILE"
fi