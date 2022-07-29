import os
import sys
os.chdir( ".." )
sys.path.append('./')

import pandas as pd
import time

from PIL import Image
import numpy as np

from data_loaders.utils import augmentImage, grabImg

DIM = 256
def generateExperimentalDataset(target_dir="datasets/synthetic_data_256/", source_dir="datasets/fft_data_256/", label_name="synthetic_2018", val_r=0.1, test_r=0.2):
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
            overlay, aug, coords = augmentImage(grabImg(source_dir+fn, DIM), DIM)
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
generateExperimentalDataset()

from data_loaders.utils import augmentImage
def emptyCSV( label_name, phase ):
    with open("data/labels/{}_{}.csv".format( label_name, phase ), "w") as lf:
        lf.write("filename,left,right,top,bottom\n")

def genDataset( source_dataset, target_dataset, label_name, target_size ):
    for phase in ["train", "test", "validation"]:
        emptyCSV( label_name, phase )

    unlabeled = pd.read_csv( "data/labels/2018_unlabeled.csv" )
    for i, fn in enumerate(unlabeled.sample(random_state=0, frac=1)["filename"]):
        if i > 0 and i % 2000 == 0:
            print(i)
            time.sleep(30) # taki doesn't like it if I create 6000 files at a time, so cap it at 2000

        if i <= len(unlabeled["filename"]) * (test_r+val_r):
            overlay, aug, coords = augmentImage(grabImg(source_dir+fn, target_size), target_size)
            img = Image.fromarray(aug.reshape((target_size,target_size)).astype(np.uint8))
            img.save(target_dir + fn)
            
            target_file = label_name+"_validation.csv"

            if i <= len(unlabeled["filename"]) * (test_r):
                target_file = label_name+"_test.csv"
        else:
            target_file = label_name+"_train.csv"
            coords = [0,0,0,0]