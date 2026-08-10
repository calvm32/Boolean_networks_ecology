#!/bin/bash

# BN_Ecology Simulation Execution Tool

set -e

# HPC Environment Setup
module purge
module load apptainer compiler/gcc/11 openmpi/4.1

# Configuration & Paths
PROJECT_DIR=$(pwd)
OUTPUT_BASE="results"
SIM_DIR="simulate/simulate_CURRENT"

# Styling & Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "BN_Ecology Simulation Execution Tool"
echo "==================================================="
echo -e "${NC}"

apptainer exec --bind "$PROJECT_DIR:$PROJECT_DIR" --pwd "$PROJECT_DIR" "$IMAGE" bash -c "pip install --user -e . && pip install --user -r requirements.txt"

# System Core Detection (Runs via Apptainer so it reads container constraints)
MAX_CORES=$(apptainer exec "$IMAGE" python3 -c "import os; print(os.cpu_count() or 1)")

# Ensure output base directory exists
mkdir -p "$OUTPUT_BASE"

# Find Runnable Scripts
mapfile -t SCRIPTS < <(find "$SIM_DIR" -type f -name "*.py" ! -name "__init__.py" ! -name "helper_funcs.py" ! -name "rules*.py" ! -name "working_params.py" | sort)

if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No python scripts found in $SIM_DIR${NC}"
    exit 1
fi

# Interactive Script Selection
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

# Parallel Execution & Core Selection
NUM_CORES=1
if [[ "$SCRIPT_NAME" == *"parallel"* ]]; then
    echo -e "\n${MAGENTA}${BOLD}⚡ Parallel Script Detected:${NC} ${BOLD}$SCRIPT_NAME${NC}"
    echo -e "  Available container CPU cores: ${BOLD}$MAX_CORES${NC}"
    read -p "$(echo -e "${CYAN}  Enter number of cores to allocate [Default: $MAX_CORES]: ${NC}")" USER_CORES
    
    if [[ -z "$USER_CORES" ]]; then NUM_CORES=$MAX_CORES;
    elif [[ "$USER_CORES" =~ ^[0-9]+$ ]] && [ "$USER_CORES" -ge 1 ] && [ "$USER_CORES" -le "$MAX_CORES" ]; then NUM_CORES=$USER_CORES;
    else NUM_CORES=$MAX_CORES; fi
else
    read -p "$(echo -e "\n${CYAN}Run with parallel workers? (Enter core count 1-$MAX_CORES, [Default: 1]): ${NC}")" USER_CORES
    if [[ "$USER_CORES" =~ ^[0-9]+$ ]] && [ "$USER_CORES" -ge 1 ] && [ "$USER_CORES" -le "$MAX_CORES" ]; then NUM_CORES=$USER_CORES; fi
fi

# Organize Output Directory & Environment
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"

mkdir -p "$RUN_OUT_DIR/data" "$RUN_OUT_DIR/figures" "$RUN_OUT_DIR/logs"

# Export variables (Apptainer automatically passes these into the container!)
export SIM_OUTPUT_DIR="$RUN_OUT_DIR"
export SIM_NUM_CORES="$NUM_CORES"
export OMP_NUM_THREADS="$NUM_CORES"
export MKL_NUM_THREADS="$NUM_CORES"
export OPENBLAS_NUM_THREADS="$NUM_CORES"

# Summary Card
echo -e "\n${GREEN}✓ Execution Environment Prepared${NC}"
echo -e "  ┌─────────────────────────────────────────────────────────┐"
echo -e "  │ ${BOLD}Script:${NC}    $SCRIPT_PATH"
echo -e "  │ ${BOLD}Cores:${NC}     $NUM_CORES / $MAX_CORES CPU core(s) allocated"
echo -e "  │ ${BOLD}Output:${NC}    $RUN_OUT_DIR"
echo -e "  └─────────────────────────────────────────────────────────┘"
echo -e "${CYAN}-----------------------------------------------------------${NC}\n"

# Execution & Logging
LOG_FILE="$RUN_OUT_DIR/logs/terminal_output.log"

cat <<EOF > "$RUN_OUT_DIR/run_info.json"
{
  "script": "$SCRIPT_PATH",
  "timestamp": "$TIMESTAMP",
  "allocated_cores": $NUM_CORES,
  "container_image": "$IMAGE",
  "user": "$(whoami)",
  "hostname": "$(hostname)"
}
EOF

echo -e "${YELLOW}Starting execution inside Apptainer... (Output streaming to log)${NC}\n"

# This is the magic line. It runs your python script *inside* the container,
# but streams the output back to your host terminal and log file.
apptainer exec \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$IMAGE" \
    python3 "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

# Completion Wrap-Up
# Note: PIPESTATUS[0] captures the exit code of the python script, not the 'tee' command
EXIT_CODE=${PIPESTATUS[0]}

echo -e "\n${CYAN}-----------------------------------------------------------${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Simulation finished successfully!${NC}"
else
    echo -e "${RED}${BOLD}✗ Simulation failed with exit code $EXIT_CODE.${NC}"
fi
echo -e "  Results directory: ${BOLD}$RUN_OUT_DIR${NC}"
echo -e "${CYAN}===========================================================${NC}\n"