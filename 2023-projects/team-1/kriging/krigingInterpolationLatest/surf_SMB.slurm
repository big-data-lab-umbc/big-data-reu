#!/bin/bash
#SBATCH --job-name=surf_SMB
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --qos=medium+
#SBATCH --output=surf_SMB-%x-%j-%u.out
#SBATCH --error=surf_SMB-%x-%j-%u.err
#SBATCH --time=05:00:00

module load Python/3.7.6-intel-2019a
pwd
source /home/adewar1/reu2023_team1/research/Angelina/kriging_interpolation/myEnv/bin/activate


python run.py surf_SMB

# Ex:
#python run.py surf_vy &
#python run.py surf_dhdt &
#python run.py surf_SMB &
#python run.py surf_elv &


wait

python --version

echo $?