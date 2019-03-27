#!/bin/bash

#SBATCH --job-name=PS-H20-SiO2
#SBATCH --account=theo1
#SBATCH --partition=alcc1
#SBATCH --mem=20000mb
#SBATCH --time=40:00:00

# parallel jobs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
export OMP_NUM_THREADS=10

echo "# starting on `hostname` at `date`"
srun /alcc/gpfs1/sw/anaconda3/bin/python compute_energy.py
echo "# finished at `date`"
