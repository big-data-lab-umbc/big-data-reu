#!/bin/bash
#SBATCH --job-name=multiout_convlstm
#SBATCH --output=slurm.multiout_convlstm_sequenced_filled.out
#SBATCH --error=slurm.multiout_convlstm_sequenced_filled.err
#SBATCH --partition=gpu2018
#SBATCH --gres=gpu:1
#SBATCH --qos=medium+
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=MaxMemPerNode
#SBATCH --account=cybertrn

module load Python/3.7.6-intel-2019a

srun python -u multiout_convlstm_sequenced_filled.py
