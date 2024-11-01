#!/bin/sh


TAKI=/umbc/xfs1
ADA=/nfs/rs

###############################
# Edit these
###############################
module load Python/3.7.6-intel-2019a  # taki, not sure if old
# module load Python/3.10.4-GCCcore-11.3.0-bare  # ada
CLSTR=$TAKI  # $TAKI or $ADA
PROGRAM_BASE=${CLSTR}/cybertrn/reu2024/team2/base   # don't change this
PIPELINE_BASE=${PROGRAM_BASE}/pp2/barajas_pp
INPUT_NAME=sharma.csv
OUTPUT_NAME=sharma
SPLIT=0.05


############################### # 
# Run pp2.py in the PIPELINE_BASE 
###############################
python ${PIPELINE_BASE}/pp2.py \
    -i ${PROGRAM_BASE}/pp1/data/${INPUT_NAME} \
    -d ${OUTPUT_NAME} \
    -s ${SPLIT} \
