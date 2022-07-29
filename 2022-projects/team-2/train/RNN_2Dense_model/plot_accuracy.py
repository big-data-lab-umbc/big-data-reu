from json import loads
from pandas import DataFrame as df
import pandas as pd
import matplotlib
from os import mkdir
from os.path import isdir
from os import listdir
from sys import argv
import numpy

def tstringToSeconds(time):
    times = time.split(":")
    seconds = float(times[-1])
    minutes = float(times[-2])
    hours   = float(times[-3])
    seconds += minutes*60 + hours * 60 * 60
    return seconds
def secondsToTime(time):
    hours   = int(time // 60 // 60)
    minutes = int((time - hours * 60 * 60) // 60)
    seconds = time - minutes*60 - hours * 60 * 60
    return "{:03d}:{:02d}:{:05.2f}".format(hours, minutes, seconds)
# Gather results from given directory
# Extract elements from lists
# infile
indata = pd.read_csv(argv[1])
epochs = indata['epoch'].to_list()
tacc1   = indata['accuracy'].to_list()
vacc1   = indata['val_accuracy'].to_list()
time  = indata['epoch_time'].to_list()
time  = sum(list(map(lambda t: tstringToSeconds(t), time)))
time  = secondsToTime(time)

import matplotlib.pyplot as plt
from numpy import linspace
fig, ax = plt.subplots()
ax.grid(b=True)
print(len(tacc1))
print(len(vacc1))
ax.plot(epochs,tacc1,label="Training")
ax.plot(epochs,vacc1,label="Validation")
ax.set_title("{} :: {}".format(max(vacc1), time))
ax.set_yticks(linspace(0,1,11))
ax.set_xlabel('epochs')
ax.set_ylabel('accuracy')
ax.set_ylim([0, 1])
ax.legend()
fig.savefig("accuracy_1.png".format())
plt.close()
