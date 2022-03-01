#!/bin/bash
#SBATCH --job-name=convlstm
#SBATCH --output=slurm.transform_%x_%j_%u.convlstm_sequenced_filled.out
#SBATCH --error=slurm.transform_%x_%j_%u.convlstm_sequenced_filled.err
#SBATCH --partition=high_mem
#SBATCH --qos=long+
#SBATCH --time=11-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=MaxMemPerNode
#SBATCH --account=cybertrn

free -g

conda activate py38-env

srun python -u transform_convlstm_sequenced_filled.py
