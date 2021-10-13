#!/bin/bash
#SBATCH --job-name=PGML
#SBATCH --account=cybertrn
#SBATCH --output=slurm.%a.out
#SBATCH --error=slurm.%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=short+
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=MaxMemPerNode
#SBATCH --array=1-2%1

if [ -d "model-final" ]
then
    scancel $SLURM_ARRAY_JOB_ID
else
    module load Python/3.7.6-intel-2019a
    mpirun python -u main.py resume_latest 
fi
