#!/bin/bash
#SBATCH --job-name=PGML
#SBATCH --account=cybertrn
#SBATCH --output=slurm.%a.out
#SBATCH --error=slurm.%a.err
#SBATCH --partition=gpu2018
#SBATCH --gres=gpu:1
#SBATCH --qos=normal+
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=35G
#SBATCH --array=1-2%1

if [ -d "model-final" ]
then
    scancel $SLURM_ARRAY_JOB_ID
else
    module load Python/3.7.6-intel-2019a
    mpirun python -u main.py resume_latest 
    python3 plot_accuracy.py training_log_00.csv
fi
