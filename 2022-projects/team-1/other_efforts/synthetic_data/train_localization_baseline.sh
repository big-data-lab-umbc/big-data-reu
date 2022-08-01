#!/bin/bash
LOGGING_ROOT="/umbc/xfs1/cybertrn/users/tchapma1/research"

CMD_ARGS="$*"

DATE=$(date +%Y%m%d-%H%M%S)
echo "srun python -u train_localization_baseline.py $CMD_ARGS $DATE"

SLURM_DIR="$LOGGING_ROOT/logs/slurm/train_localization_baseline/"
[ -d $SLURM_DIR ] || mkdir $SLURM_DIR

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=train_localization_baseline
#SBATCH --output=$LOGGING_ROOT/logs/slurm/train_localization_baseline/$DATE.out
#SBATCH --error=$LOGGING_ROOT/logs/slurm/train_localization_baseline/$DATE.err
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:1
#SBATCH --qos=normal+
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30G
#SBATCH --account=cybertrn

module load Python/3.7.6-intel-2019a

srun python -u train_localization_baseline.py $CMD_ARGS $DATE
EOT