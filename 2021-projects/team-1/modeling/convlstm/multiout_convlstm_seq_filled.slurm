#!/bin/bash
#SBATCH --job-name=multiout_convlstm
#SBATCH --output=slurm_%x_%j_%u_multiout_convlstm_sequenced_filled.out
#SBATCH --error=slurm_%x_%j_%u_multiout_convlstm_sequenced_filled.err
#SBATCH --partition=gpu2018
#SBATCH --gres=gpu:1
#SBATCH --qos=medium+
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=MaxMemPerNode
#SBATCH --account=cybertrn

conda activate py38-env

srun python -u ieee_multiout_convlstm_sequenced_filled.py
