#!/bin/bash
# Usage (once, on the login node):   python sobol_parallel.py prepare
# Then:                               sbatch run_sobol.sbatch
#
# Short, small jobs backfill into scheduler gaps much faster than one huge request.
# If the queue is still slow, try e.g.  --array=0-63 --cpus-per-task=4 --time=01:30:00
# Everything is checkpointed per chunk, so timeouts/preemption are harmless: just resubmit.
#
#SBATCH --job-name=sobol
#SBATCH --array=0-15
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/sobol_%A_%a.out

mkdir -p logs

# ---- environment: edit for your cluster ----
# module load python/3.11
# source ~/venvs/sobol/bin/activate
# export SIM_OUTPUT_DIR=$SCRATCH/sobol_run    # optional; must be on a filesystem all array tasks share

# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT / SLURM_CPUS_PER_TASK are picked up automatically.
python sobol_parallel.py run
