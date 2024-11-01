#!/bin/bash
#SBATCH --job-name=pjin8.1_barajas_4layer                                 
#SBATCH --mem=48G                                   
#SBATCH --nodes=1               # num nodes: for now, we are doing all prediction runs at 1
#SBATCH --gres=gpu:1            # num gpus per node: for now, we are doing all prediction runs at 1
#SBATCH --ntasks-per-node=1     # num gpus per node: for now, we are doing all prediction runs at 1
#SBATCH --time=0-3:00:00
#SBATCH --constraint=rtx_6000   # for now, we are doing all prediction runs on an rtx_6000 node
#SBATCH --error=slurm_output/slurm_pred.err
#SBATCH --output=slurm_output/slurm_pred.out

# variables
run_id='pjin8.1_barajas_4layer'  # CHANGE THIS
# shouldn't change variables below
PROGRAM_BASE=/nfs/rs/cybertrn/reu2024/team2/base
config_path='../../config/'
config_path+=${run_id}
config_path+='.yaml'

# DON'T CHANGE ANYTHING BELOW
# activate conda env
module load Anaconda3/2023.09-0
source /usr/ebuild/software/Anaconda3/2023.09-0/bin/activate
echo "activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /nfs/rs/cybertrn/reu2024/team2/envs/ada_main  # choose which environment carefully  
echo "conda environment activated."

# debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# # write this script to the outputsß
# scontrol write bash_script $SLURM_JOB_ID slurm_output/script.txt

# run
srun python3 ../../predict.py -c $config_path
echo "finished running predict.py, now running eval.py ..."
# get the largest version number
cd ${PROGRAM_BASE}/logs/csv_logs/${run_id}/lightning_logs/
version=$(ls | grep '^version_[0-9]\+$' | cut -c2- | sort -n | tail -n1 | sed 's|^|v|')
echo "using version: "${version}
cd - 
# do rest of run
srun python3 ../../eval/eval.py -o b -p eval/${run_id}/ -t logs/csv_logs/${run_id}/lightning_logs/${version}/
echo "finished running eval.py"
conda deactivate
echo "conda environment deactivated."
