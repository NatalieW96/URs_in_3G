#!/bin/bash
#SBATCH -J MM_BL
#SBATCH -o MM_BL_logs/output.log
#SBATCH -e MM_BL_logs/error.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=32
#SBATCH -p cpu

# Set thread limits for parallel processing
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Run the Python script with all posterior files
srun -u python calculate_mismatches_BL.py
