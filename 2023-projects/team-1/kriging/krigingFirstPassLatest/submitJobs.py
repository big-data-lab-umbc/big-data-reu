import os
import numpy as np
'''
RANGE = 632_612
STEP = 10_000
'''
START = 600_000
RANGE = 632_612
STEP = 5_000
for n in np.arange(START, RANGE - STEP, STEP):
    start = n
    stop = n + STEP
    cmd = 'sbatch run.slurm ' + str(start) + ' ' + str(stop)
    os.system(cmd)

# Handle the left over too small batch if needed
if (n + STEP < RANGE):
    start = n + STEP
    stop = RANGE
    cmd = 'sbatch run.slurm ' + str(start) + ' ' + str(stop)
    os.system(cmd)

print("All jobs submitted to taki.")