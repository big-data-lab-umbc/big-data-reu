#!/bin/bash
# args are model_name dataset labelset num_epochs
# e.g. bash meta_slurm.sh pure_localization_inceptionv3 fft_data_256 2018 200

CMD_ARGS="$*"

MODEL=$1
DATASET=$2
LABEL_SET=$3
EPOCHS=$4

DATE=$(date +%Y%m%d-%H%M%S)
echo "srun python -u meta_train.py $CMD_ARGS $DATE"

SLURM_DIR="/umbc/xfs1/cybertrn/users/tchapma1/research/logs/slurm/$MODEL-$DATASET-$LABEL_SET-$EPOCHS/"
[ -d $SLURM_DIR ] || mkdir $SLURM_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=Theo_$MODEL_$EPOCHS
#SBATCH --output=/umbc/xfs1/cybertrn/users/tchapma1/research/logs/slurm/$MODEL-$DATASET-$LABEL_SET-$EPOCHS/$DATE.out
#SBATCH --error=/umbc/xfs1/cybertrn/users/tchapma1/research/logs/slurm/$MODEL-$DATASET-$LABEL_SET-$EPOCHS/$DATE.err
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --qos=medium+
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --account=cybertrn

module load Python/3.7.6-intel-2019a

srun python -u meta_train.py $CMD_ARGS $DATE
EOT