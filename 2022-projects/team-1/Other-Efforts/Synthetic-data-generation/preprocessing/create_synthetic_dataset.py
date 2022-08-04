import os
import sys
os.chdir( ".." )
sys.path.append('./')

import pandas as pd
import time

from PIL import Image
import numpy as np

from data_loaders.gravity_wave_generation.utils import augmentImage, grabImage
from utils.paths import DATASET_PATHS

### CHANGE THESE TO MAKE A DATASET FOR A NEW GENERATION METHOD
from data_loaders.gravity_wave_generation.generate_gravity_wave_simple import generateWavePattern
dataset_name = "simple_synthetic_data_256"
source_dataset = "fft_data_256"
label_name = "simple_synthetic_2018"
val_r = 0.1
test_r = 0.2
image_dim = 256
### END PARAMETERS

try:
    dest = DATASET_PATHS[dataset_name]
except:
    raise Exception( "The path to {} needs to be defined in utils.paths.DATASET_PATHS".format(dataset_name) )
try:
    source = DATASET_PATHS[source_dataset]
except:
    raise Exception( "The path to {} needs to be defined in utils.paths.DATASET_PATHS".format(source_dataset) )

def generateExperimentalDataset(target_dir, source_dir, label_name, val_r, test_r, DIM):
    with open("data_loaders/labels/"+label_name+"_train.csv", "w") as lf:
        lf.write("filename,left,top,right,bottom\n")
    with open("data_loaders/labels/"+label_name+"_validation.csv", "w") as lf:
        lf.write("filename,left,top,right,bottom\n")
    with open("data_loaders/labels/"+label_name+"_test.csv", "w") as lf:
        lf.write("filename,left,top,right,bottom\n")
    unlabeled = pd.read_csv( "data_loaders/labels/2018_unlabeled.csv" )

    for i, fn in enumerate(unlabeled.sample(random_state=0, frac=1)["filename"]):
        if i > 0 and i % 2000 == 0:
            print(i)
            time.sleep(30) # taki doesn't like it if I create 6000 files at a time, so cap it at 2000

        if i <= len(unlabeled["filename"]) * (test_r+val_r):
            overlay, aug, coords = augmentImage(grabImage(source_dir+fn, DIM), DIM, generateWavePattern)
            img = Image.fromarray(aug.reshape((DIM,DIM)).astype(np.uint8))
            img.save(target_dir + fn)
            
            target_file = label_name+"_validation.csv"

            if i <= len(unlabeled["filename"]) * (test_r):
                target_file = label_name+"_test.csv"
        else:
            target_file = label_name+"_train.csv"
            coords = [0,0,0,0]

        with open("data_loaders/labels/"+target_file, "a") as lf:
            lf.write(",".join([fn]+list([str(x) for x in coords]))+"\n")
generateExperimentalDataset(dest, source, label_name, val_r, test_r, image_dim)