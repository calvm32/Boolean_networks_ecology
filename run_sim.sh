#!/bin/bash
# Boolean Networks Ecology Simulation Execution Tool

set -e

# Configuration & Paths
PROJECT_DIR=$(pwd)
OUTPUT_BASE="results"
SIM_DIR="simulate/simulate_CURRENT"
VENV_DIR="$HOME/envs/bn_ecology_env"

# Styling & Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "Boolean Networks Ecology Simulation Execution Tool"
echo "==================================================="
echo -e "${NC}"

# Environment Setup (Runs ONCE)
# This checks if the environment exists. If not, it builds it.
# It will skip this entirely on future runs.

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}First time setup detected: Creating Python virtual environment...${NC}"
    module purge
    module load python/3.10
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Environment created successfully!${NC}\n"
fi

MAX_CORES=$(nproc 2> /dev/null || echo 64)
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

echo -e "\n${CYAN}${BOLD}⚡ Resource Configuration${NC}"

read -p "  Enter number of cores to allocate (1-$MAX_CORES) [Default: 1]: " USER_CORES
NUM_CORES=${USER_CORES:-1}
if ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]] || [ "$NUM_CORES" -lt 1 ]; then
    NUM_CORES=1
fi

read -p "  Enter maximum walltime (HH:MM:SS) [Default: 12:00:00]: " USER_TIME
WALLTIME=${USER_TIME:-12:00:00}

MEM_GB=$((NUM_CORES * 4))

# Organization

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"
mkdir -p "$RUN_OUT_DIR/data" "$RUN_OUT_DIR/figures" "$RUN_OUT_DIR/logs"

# Generate SLURM Batch Script
# This is where the #SBATCH tags actually get written into a new file,
# ensuring the exact parameters are saved in your results folder forever.

BATCH_FILE="$RUN_OUT_DIR/submit_${SCRIPT_NAME}.sh"

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

# Setup compute node environment
module purge
module load python/3.10 gcc/11

# Activate the virtual environment we created earlier
source "${VENV_DIR}/bin/activate"

# Export variables for Python multiprocessing
export SIM_OUTPUT_DIR="${RUN_OUT_DIR}"
export SIM_NUM_CORES="${NUM_CORES}"
export OMP_NUM_THREADS="${NUM_CORES}"
export MKL_NUM_THREADS="${NUM_CORES}"
export OPENBLAS_NUM_THREADS="${NUM_CORES}"

echo "==================================================="
echo "Starting Execution: ${SCRIPT_NAME}"
echo "==================================================="

python3 "${PROJECT_DIR}/${SCRIPT_PATH}"
EOF

# Generate JSON Run Record
cat <<EOF > "$RUN_OUT_DIR/run_info.json"
{
  "script": "$SCRIPT_PATH",
  "timestamp": "$TIMESTAMP",
  "allocated_cores": $NUM_CORES,
  "walltime": "$WALLTIME",
  "memory_gb": $MEM_GB,
  "user": "$(whoami)"
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
    echo -e "${YELLOW}Running directly in current Slurm allocation...${NC}"
    bash "$BATCH_FILE"
fi