#!/bin/bash
#SBATCH --job-name=gpr_full
#SBATCH --partition=batch
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --constraint=hpcf2013
#SBATCH --qos=medium+
#SBATCH --mem=MAX
#SBATCH --output=slurm-%x-%j-%u.out
#SBATCH --error=slurm-%x-%j-%u.err
#SBATCH --time=6:00:00

module load Python/3.8.2-GCCcore-9.3.0
mpirun python3 gpr_for1201Full_ky_230627.py

EXIT_STATUS=$?

export CONDA_PREFIX=/umbc/xfs1/cybertrn/common/Softwares/anaconda3/
export PYSPARK_PYTHON=$CONDA_PREFIX/bin/python
export PYSPARK_DRIVER_PYTHON=$CONDA_PREFIX/bin/python
export PYTHONPATH=$CONDA_PREFIX/bin:$PYTHONPATH
export PATH=$CONDA_PREFIX/bin:$PATH
SPARK=/umbc/xfs1/cybertrn/common/Softwares/spark/spark-2.4.0-bin-hadoop2.7
MY_SPARK=$(pwd)
SPARK_PYTHON_FILE=$SPARK/examples/src/main/python/pi.py
SPARK_PYTHON_ARGUMENTS=100

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
echo "Current Time : $current_time"
MY_SPARK=$MY_SPARK/$SLURM_JOB_ID-$current_time
mkdir -p $MY_SPARK
EXE_LOG_PATH_MY_SPARK=$MY_SPARK/logs
cp -r $SPARK/conf $MY_SPARK
export SPARK_CONF_DIR=$MY_SPARK/conf
mkdir -p $EXE_LOG_PATH_MY_SPARK


#Step 2: Update slaves file at $SPARK/conf based on the nodes allocated from job scheduler.
cat /dev/null > $MY_SPARK/conf/slaves
master=$(echo $SLURMD_NODENAME)

nodes=$(echo $SLURM_NODELIST | cut -d'[' -f 2 | cut -d']' -f 1)
echo "nodes: $nodes"
echo $(echo $nodes| cut -d'-' -f 2)
if [[ $nodes == *","* ]]; then
  for element in ${nodes//,/ } ; do
    echo "element:$element"
    if [[ $element == *"-"* ]]; then
      start_node=$(echo $element| cut -d'-' -f 1)
      end_node=$(echo $element| cut -d'-' -f 2)
      #echo "start_node:$start_node, end_node:$end_node"
      for sub_element in $(seq -f "%03g" $start_node $end_node) ; do
        if [ "cnode$sub_element" != "$master" ]; then
          echo 'cnode'$sub_element >> $MY_SPARK/conf/slaves
        fi
      done
    else
      if [ "cnode$element" != "$master" ]; then
        echo 'cnode'$element >> $MY_SPARK/conf/slaves
      fi
    fi
  done
else
  start_node=$(echo $nodes| cut -d'-' -f 1)
  end_node=$(echo $nodes| cut -d'-' -f 2)
  for sub_element in $(seq -f "%03g" $start_node $end_node) ; do
    if [ "cnode$sub_element" != "$master" ]; then
          echo 'cnode'$sub_element >> $MY_SPARK/conf/slaves
    fi
  done
fi  

echo "slaves: $(cat $MY_SPARK/conf/slaves)"

echo $(egrep --color 'Mem|Cache|Swap' /proc/meminfo)
echo $(ulimit -a)



#Step 3: Start/deploy Spark on all nodes allocated
$SPARK/sbin/stop-all.sh
sleep 5

ulimit -c unlimited
#$SPARK/sbin/start-master.sh
$SPARK/sbin/start-all.sh
sleep 5

host=$(hostname)


#Step 4: Submit your Spark job and wait for its finish
echo "time $SPARK/bin/spark-submit --master spark://$master:7077 --driver-memory 60g --executor-memory 4g $SPARK_PYTHON_FILE" $SPARK_PYTHON_ARGUMENTS

echo "time $SPARK/bin/spark-submit --master spark://$master:7077 --driver-memory 60g --executor-memory 4g $SPARK_PYTHON_FILE" $SPARK_PYTHON_ARGUMENTS > $EXE_LOG_PATH_MY_SPARK/$current_time-script-log.txt

(time $SPARK/bin/spark-submit --master spark://$master:7077 --driver-memory 60g --executor-memory 60g $SPARK_PYTHON_FILE $SPARK_PYTHON_ARGUMENTS > $EXE_LOG_PATH_MY_SPARK/$current_time-log.txt) 2> $EXE_LOG_PATH_MY_SPARK/$current_time-time.txt

#print complete statement and stop running for failure or successful finish
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Python script executed successfully."
  scancel $SLURM_JOB_ID
else
  echo "Python script encountered an error. Exiting job."
  scancel $SLURM_JOB_ID
  exit $EXIT_STATUS
fi

#Step 5: Stop Spark at the end
sleep 5
$SPARK/sbin/stop-all.sh

