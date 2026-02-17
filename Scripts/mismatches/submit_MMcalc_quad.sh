#!/bin/bash
#SBATCH -J MM_quad
#SBATCH -o MM_quad_logs/output.log
#SBATCH -e MM_quad_logs/error.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=128
#SBATCH -p cpu

# Set thread limits for parallel processing
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Run the Python script with all posterior files
srun -u python calculate_mismatches_quad.py
