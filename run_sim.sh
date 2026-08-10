#!/bin/bash

# BN_Ecology Simulation Execution Tool
# Publication-quality output management & multi-core execution controller

module purge
module load apptainer compiler/gcc/11 openmpi/4.1

set -e

# Styling & Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD="python"
BASE_DIR=$(pwd)
SIM_DIR="simulate/simulate_CURRENT"  # or simulate/simulate_OLD
OUTPUT_BASE="results"

# System Core Detection
detect_max_cores() {
    if command -v nproc &>/dev/null; then
        nproc
    elif command -v sysctl &>/dev/null; then
        sysctl -n hw.ncpu
    else
        $PYTHON_CMD -c "import os; print(os.cpu_count() or 1)"
    fi
}

MAX_CORES=$(detect_max_cores)

# Ensure output base directory exists
mkdir -p "$OUTPUT_BASE"

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "       BN_Ecology Simulation Execution Tool        "
echo "==================================================="
echo -e "${NC}"

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

# Auto-detect if file is intended for parallel execution or prompt user
if [[ "$SCRIPT_NAME" == *"parallel"* ]]; then
    echo -e "\n${MAGENTA}${BOLD}⚡ Parallel Script Detected:${NC} ${BOLD}$SCRIPT_NAME${NC}"
    echo -e "  Available system CPU cores: ${BOLD}$MAX_CORES${NC}"
    
    read -p "$(echo -e "${CYAN}  Enter number of cores to allocate [Default: $MAX_CORES]: ${NC}")" USER_CORES
    
    if [[ -z "$USER_CORES" ]]; then
        NUM_CORES=$MAX_CORES
    elif [[ "$USER_CORES" =~ ^[0-9]+$ ]] && [ "$USER_CORES" -ge 1 ] && [ "$USER_CORES" -le "$MAX_CORES" ]; then
        NUM_CORES=$USER_CORES
    else
        echo -e "${YELLOW}  Invalid entry. Defaulting to $MAX_CORES cores.${NC}"
        NUM_CORES=$MAX_CORES
    fi
else
    # Optional override for standard scripts
    read -p "$(echo -e "\n${CYAN}Run with parallel workers? (Enter core count 1-$MAX_CORES, [Default: 1]): ${NC}")" USER_CORES
    if [[ "$USER_CORES" =~ ^[0-9]+$ ]] && [ "$USER_CORES" -ge 1 ] && [ "$USER_CORES" -le "$MAX_CORES" ]; then
        NUM_CORES=$USER_CORES
    fi
fi

# Organize Output Directory & Environment
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"

mkdir -p "$RUN_OUT_DIR/data"
mkdir -p "$RUN_OUT_DIR/figures"
mkdir -p "$RUN_OUT_DIR/logs"

# Export environment variables for Python & underlying numeric libraries
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

# Save metadata file inside the run directory for complete auditability
cat <<EOF > "$RUN_OUT_DIR/run_info.json"
{
  "script": "$SCRIPT_PATH",
  "timestamp": "$TIMESTAMP",
  "allocated_cores": $NUM_CORES,
  "max_system_cores": $MAX_CORES,
  "user": "$(whoami)",
  "hostname": "$(hostname)"
}
EOF

echo -e "${YELLOW}Starting execution... (Output streaming to terminal and log file)${NC}\n"

# Execute script with live terminal output + logging
$PYTHON_CMD "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

# Completion Wrap-Up
EXIT_CODE=${PIPESTATUS[0]}

echo -e "\n${CYAN}-----------------------------------------------------------${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Simulation finished successfully!${NC}"
else
    echo -e "${RED}${BOLD}✗ Simulation failed with exit code $EXIT_CODE.${NC}"
fi
echo -e "  Results directory: ${BOLD}$RUN_OUT_DIR${NC}"
echo -e "${CYAN}===========================================================${NC}\n"