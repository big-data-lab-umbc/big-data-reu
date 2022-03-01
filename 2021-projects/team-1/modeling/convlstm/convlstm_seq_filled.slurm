#!/bin/bash
#SBATCH --job-name=convlstm
#SBATCH --output=slurm.convlstm_sequenced_filled.out
#SBATCH --error=slurm.convlstm_sequenced_filled.err
#SBATCH --partition=high_mem
#SBATCH --qos=medium+
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=MaxMemPerNode
#SBATCH --account=cybertrn

module load Python/3.7.6-intel-2019a

srun python -u convlstm_sequenced_filled.py
