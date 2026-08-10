#!/bin/bash

# Boolean_networks_ecology Simulation Runner
# Organizes outputs into timestamped folders

# Styling & Colors
BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD="python"
BASE_DIR=$(pwd)
SIM_DIR="simulate/simulate_distribution_based"
OUTPUT_BASE="results"

# Ensure output base directory exists
mkdir -p "$OUTPUT_BASE"

echo -e "${CYAN}${BOLD}"
echo "==================================================="
echo "       BN_Ecology Simulation Execution Tool        "
echo "==================================================="
echo -e "${NC}"

# Find Runnable Scripts
# Dynamically find python files in the relevant directories, ignoring helpers
mapfile -t SCRIPTS < <(find "$SIM_DIR/compare_regimes" "$SIM_DIR/running_solvers" -type f -name "*.py")

if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No python scripts found in $SIM_DIR/compare_regimes or $SIM_DIR/running_solvers${NC}"
    exit 1
fi

# Interactive Menu
echo -e "${YELLOW}Select a script to run:${NC}"
PS3=$'\n\033[1;36mEnter the number of the script: \033[0m'

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

# Organize Output Directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_OUT_DIR="${OUTPUT_BASE}/${TIMESTAMP}_${SCRIPT_NAME}"

# Create the folder structure for this specific run
mkdir -p "$RUN_OUT_DIR/data"
mkdir -p "$RUN_OUT_DIR/figures"
mkdir -p "$RUN_OUT_DIR/logs"

# Export the directory so Python can access it via os.environ
export SIM_OUTPUT_DIR="$RUN_OUT_DIR"

echo -e "\n${GREEN}✓ Environment prepared${NC}"
echo -e "  Script:    ${BOLD}$SCRIPT_PATH${NC}"
echo -e "  Output:    ${BOLD}$RUN_OUT_DIR${NC}"
echo -e "${CYAN}---------------------------------------------------${NC}\n"

# Execution & Logging
LOG_FILE="$RUN_OUT_DIR/logs/terminal_output.log"

echo -e "${YELLOW}Starting simulation... (Output is being logged)${NC}\n"

# Run the script. use 'tee' so you can see it live AND save it to the log.
# 2>&1 redirects standard error to standard out, so errors are logged too.
$PYTHON_CMD "$SCRIPT_PATH" 2>&1 | tee "$LOG_FILE"

# Wrap Up
EXIT_CODE=${PIPESTATUS[0]}

echo -e "\n${CYAN}---------------------------------------------------${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Run completed successfully!${NC}"
else
    echo -e "${RED}${BOLD}✗ Run failed with exit code $EXIT_CODE.${NC}"
fi
echo -e "Check ${BOLD}$RUN_OUT_DIR${NC} for data, figures, and logs."
echo -e "${CYAN}===================================================${NC}\n"
