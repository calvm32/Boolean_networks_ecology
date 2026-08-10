#!/bin/bash
# Boolean Networks Ecology Simulation Execution Tool (Pure Python)

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

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}First time setup detected: Creating Python virtual environment...${NC}"
    module purge
    # MINIMAL CHANGE: Added openmpi so mpi4py compiles against the cluster hardware
    module load python/3.10 openmpi compiler/gcc/11 2>/dev/null || module load python/3.10 openmpi
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    # MINIMAL CHANGE: Added --no-cache-dir so pip doesn't reuse the broken, cached mpi4py install
    pip install --no-cache-dir -r requirements.txt
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

# MINIMAL CHANGE: Split cores into nodes and tasks for correct MPI execution
read -p "  Enter number of nodes [Default: 1]: " USER_NODES
NUM_NODES=${USER_NODES:-1}

read -p "  Enter number of tasks per node [Default: 1]: " USER_TASKS
NUM_TASKS=${USER_TASKS:-1}

read -p "  Enter maximum walltime (HH:MM:SS) [Default: 12:00:00]: " USER_TIME
WALLTIME=${USER_TIME:-12:00:00}

MEM_GB=$((NUM_TASKS * 4))

# Organization

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"
mkdir -p "$RUN_OUT_DIR/data" "$RUN_OUT_DIR/figures" "$RUN_OUT_DIR/logs"

BATCH_FILE="$RUN_OUT_DIR/submit_${SCRIPT_NAME}.sh"

# Generate SLURM Batch Script

cat <<EOF > "$BATCH_FILE"
#!/bin/bash
#SBATCH --job-name=${SCRIPT_NAME}
#SBATCH --output=${RUN_OUT_DIR}/logs/job.out
#SBATCH --error=${RUN_OUT_DIR}/logs/job.err
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=${NUM_TASKS}
#SBATCH --time=${WALLTIME}
#SBATCH --mem=${MEM_GB}G

# Setup compute node environment
module purge
# MINIMAL CHANGE: Ensured openmpi is loaded on the compute node
module load python/3.10 openmpi compiler/gcc/11 2>/dev/null || module load python/3.10 openmpi

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Add top-level project directory to Python search path
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH}"

# Export variables for Python multiprocessing (keeping variables for backwards compatibility)
export SIM_OUTPUT_DIR="${RUN_OUT_DIR}"
export SIM_NUM_CORES="${NUM_TASKS}"
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1"
export OPENBLAS_NUM_THREADS="1"

echo "==================================================="
echo "Starting Execution: ${SCRIPT_NAME}"
echo "==================================================="

# MINIMAL CHANGE: Added srun so Slurm initializes the MPI environment across all requested nodes
srun python3 "${PROJECT_DIR}/${SCRIPT_PATH}"
EOF

# Generate JSON Run Record
cat <<EOF > "$RUN_OUT_DIR/run_info.json"
{
  "script": "$SCRIPT_PATH",
  "timestamp": "$TIMESTAMP",
  "allocated_nodes": $NUM_NODES,
  "tasks_per_node": $NUM_TASKS,
  "walltime": "$WALLTIME",
  "memory_gb": $MEM_GB,
  "user": "$(whoami)"
}
EOF

# Queue Submission

echo -e "\n${GREEN}✓ Execution Environment Prepared${NC}"
echo -e "==================================================="
echo -e "   ${BOLD}Script:${NC}    $SCRIPT_PATH"
echo -e "   ${BOLD}Resources:${NC} $NUM_NODES node(s), $NUM_TASKS task(s)/node, $MEM_GB GB RAM, $WALLTIME limit"
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