#!/bin/bash
#SBATCH --job-name=accessJobCollocation
#SBATCH --account=cybertrn
#SBATCH --partition=high_mem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --qos=long+
#SBATCH --mem=MaxMemPerNode
#SBATCH --time=11-24:00:00
#SBATCH --output=./slurm_log/slurm-%x-%j-%u.out
#SBATCH --error=./slurm_log/slurm-%x-%j-%u.err

free -h
module load Python/3.7.6-intel-2019a
python multiout_convlstm_sequenced_filled.py