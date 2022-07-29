#!/bin/bash
#SBATCH --job-name=PGML
#SBATCH --account=cybertrn
#SBATCH --output=slurm.%a.out
#SBATCH --error=slurm.%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --qos=normal+
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=MaxMemPerNode
#SBATCH --array=1-4%1

if [ -d "model-final" ]
then
    scancel $SLURM_ARRAY_JOB_ID
else
    module load Python/3.7.6-intel-2019a
    mpirun python -u main.py resume_latest
    python3 plot_accuracy.py training_log_00.csv 
fi
