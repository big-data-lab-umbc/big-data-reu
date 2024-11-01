#!/bin/bash
#SBATCH --job-name=pjin7.22_mothership                        # Job name
#SBATCH --mem=48G                                   # Job memory request
#SBATCH --nodes=1                                   # num nodes
#SBATCH --gres=gpu:4                                # num gpus per node
#SBATCH --ntasks-per-node=4   
#SBATCH --time=3-23:00:00                           # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000                       # Specific hardware constraint
#SBATCH --error=slurm_output/slurm.err    # Error file name
#SBATCH --output=slurm_output/slurm.out   # Output file name

# variables
run_id='pjin7.22_mothership'
file_name='../../train.py'
config_path='../../config/'  # don't forget to change this!
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

# # write this script to the output
# scontrol write bash_script $SLURM_JOB_ID slurm_output/${run_id}/script.txt

# run
srun python3 $file_name -c $config_path
conda deactivate
echo "conda environment deactivated."
