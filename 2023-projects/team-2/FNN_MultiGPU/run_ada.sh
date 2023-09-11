#!/bin/bash
#SBATCH --job-name=PL_FF            # Job name
#SBATCH --mem=48G                      # Job memory request
#SBATCH --gres=gpu:1                   # Number of requested GPU(s)
#SBATCH --time=3-23:00:00                  # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000           # Specific hardware constraint
#SBATCH --error=slurm.err               # Error file name
#SBATCH --output=slurm.out              # Output file name

if [ -f "model-final" ] || [ -d "model-final" ]
then
    scancel $SLURM_ARRAY_JOB_ID
else
    
    module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
    module load Anaconda3/2020.07
    source /usr/ebuild/software/Anaconda3/2020.07/bin/activate
    conda activate /nfs/rs/cybertrn/reu2023/team2/research/ada_envs/torch-env

    # debugging flags
    export NCCL_DEBUG=INFO
    export PYTHONFAULTHANDLER=1

    srun python main.py resume_training
fi
