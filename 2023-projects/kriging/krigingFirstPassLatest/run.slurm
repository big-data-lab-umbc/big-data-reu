#!/bin/bash
#SBATCH --job-name=kriging_fixed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --qos=medium+
#SBATCH --output=slurm-%x-%j-%u.out
#SBATCH --error=slurm-%x-%j-%u.err
#SBATCH --time=03:30:00

module load Python/3.7.6-intel-2019a
cd /home/adewar1/reu2023_team1/research/Angelina/newKrigingMethodAD
pwd
source /home/adewar1/reu2023_team1/research/Angelina/newKrigingMethodAD/myEnv/bin/activate


python krigingfirstpass_improvingaccuracyattempt.py $1 $2

wait
#632612
python --version

echo $?