#!/bin/bash
# args are model_name dataset num_epochs
# e.g. bash meta_slurm.sh basic_cnn labeled_data 100

CMD_ARGS="$*"
# echo $CMD_ARGS

# NAME=${1%.*}
# EPOCHS=$2

MODEL=$1
DATASET=$2
EPOCHS=$3

DATE=$(date +%Y%m%d-%H%M%S)
# echo $DATE
echo "srun python -u meta_train.py $CMD_ARGS $DATE"

# NEW_DIR_A="/umbc/xfs1/cybertrn/users/tchapma1/research/logs/training/$MODEL-$DATASET-$EPOCHS/"
# NEW_DIR_B="/umbc/xfs1/cybertrn/users/tchapma1/research/logs/training/$MODEL-$DATASET-$EPOCHS/$DATE/"

# echo $NEW_DIR_A
# echo $NEW_DIR_B
# [ -d $NEW_DIR_A ] || mkdir $NEW_DIR_A
# [ -d $NEW_DIR_B ] || mkdir $NEW_DIR_B

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=QuickRunPythonScript
#SBATCH --output=/home/kchen/reu2022_team1/research/autoencoder/best_autoencoder_3.out
#SBATCH --error=/home/kchen/reu2022_team1/research/autoencoder/best_autoencoder_3.err
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --qos=normal+
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --account=cybertrn

module load Python/3.7.6-intel-2019a

srun python -u meta_train.py $CMD_ARGS $DATE
EOT
