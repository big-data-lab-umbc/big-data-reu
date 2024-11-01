#!/bin/bash
#SBATCH --job-name=pred                         # Job name
#SBATCH --mem=48G                                   # Job memory request
#SBATCH --nodes=1                                   # num nodes
#SBATCH --gres=gpu:1                                # num gpus per node
#SBATCH --ntasks-per-node=1   
#SBATCH --time=3-23:00:00                           # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000                       # Specific hardware constraint
#SBATCH --error=slurm_output/slurm_pred.err    # Error file name
#SBATCH --output=slurm_output/slurm_pred.out   # Output file name

# variables
run_id='mc7.15_FCNv1'
file_name='../../../predict.py'
config_path='../../../config/jul15/'
config_path+=${run_id}
config_path+='.yaml'

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
srun python3 $file_name -c $config_path
conda deactivate
echo "conda environment deactivated."
