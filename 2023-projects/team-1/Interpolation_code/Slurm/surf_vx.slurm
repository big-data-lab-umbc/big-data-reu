#!/bin/bash
#SBATCH --job-name=surf_vx
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --qos=medium+
#SBATCH --output=surf_vx-%x-%j-%u.out
#SBATCH --error=surf_vx-%x-%j-%u.err
#SBATCH --time=05:00:00

module load Python/3.7.6-intel-2019a
pwd
source /home/adewar1/reu2023_team1/research/Angelina/kriging_interpolation/myEnv/bin/activate


python UK_Interpolation.py surf_vx

# Ex:
#python UK_Interpolation.py  surf_vy &
#python UK_Interpolation.py  surf_dhdt &
#python UK_Interpolation.py  surf_SMB &
#python UK_Interpolation.py  surf_elv &


wait

python --version

echo $?
