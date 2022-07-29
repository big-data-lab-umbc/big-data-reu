#!/bin/bash
#SBATCH --job-name=RNRB             # Job name
#SBATCH --mail-user=jclark16@umbc.edu      # Where to send mail
#SBATCH --mem=30000                      # Job memory request
#SBATCH --gres=gpu:4                    # Number of requested GPU(s) 
#SBATCH --time=5-20:10:00                  # Time limit days-hrs:min:sec
#SBATCH --constraint=rtx_6000           # Specific hardware constraint
#SBATCH --error=slurm.err               # Error file name
#SBATCH --output=slurm.out              # Output file name

module load Anaconda3/2020.07  # load up the correct modules, if required
module load TensorFlow/2.6.0-foss-2021a-CUDA-11.3.1
python -u main.py                # launch the code
