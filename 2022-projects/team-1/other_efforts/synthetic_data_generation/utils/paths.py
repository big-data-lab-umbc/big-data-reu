## relative paths are relative to the parent directory of this file
USER_LOGS = "../logs/"
LOG_DIR_NAME = "synthetic_data/" # except this one, which is relative to USER_LOGS
# where are the hdf5 files
HDF5_ROOT = "/umbc/xfs1/cybertrn/reu2022/team1/datasets/hdf5/vnb_patch_data_2018/"

DATASET_PATHS = {
    "data_256": "/umbc/xfs1/cybertrn/reu2022/team1/datasets/data_256/",
    "fft_data_256": "/umbc/xfs1/cybertrn/reu2022/team1/datasets/fft_data_256/",
    "synthetic_data_256": "/umbc/xfs1/cybertrn/reu2022/team1/datasets/synthetic_data_256/",
    "simple_synthetic_data_256": "/umbc/xfs1/cybertrn/reu2022/team1/datasets/simple_synthetic_data_256/",
    "2020_1000": "/umbc/xfs1/cybertrn/reu2022/team1/datasets/2020_1000/",
}

# "dataset_name" : which dataset contains the unaugmented images you want to use for training the localization model and doing transfer,
# "test_dataset_name" : which dataset contains the val and test images for the localization model,
# "label_set" : which label set do you want to train the localization model on,
# "transfer_label_set" : which label set do you want to train the transfer model on,
# "batch_generator" : custom_get_batches( the wave pattern generator you want to use )
from data_loaders.gravity_wave_generation.utils import custom_get_batches
from data_loaders.gravity_wave_generation import generate_gravity_wave_simple, generate_gravity_wave_v2
DIM = 256
MODE_DICTIONARY = {
    "simple": {
        "dataset_name" : "fft_data_{}".format(DIM),
        "test_dataset_name" : "simple_synthetic_data_{}".format(DIM),
        "label_set" : "simple_synthetic_2018",
        "transfer_label_set" : "2018",
        "batch_generator" : custom_get_batches( generate_gravity_wave_simple.generateWavePattern )
    },
    "default": {
        "dataset_name" : "fft_data_{}".format(DIM),
        "test_dataset_name" : "synthetic_data_{}".format(DIM),
        "label_set" : "synthetic_2018",
        "transfer_label_set" : "2018",
        "batch_generator" : custom_get_batches( generate_gravity_wave_v2.generateWavePattern )
    }
}

import os
import datetime

def getDataDirectory( dataset_name ):
    return DATASET_PATHS[dataset_name]

def getTrainingLogRoot(name):
    fp = USER_LOGS + "{}/{}/".format(LOG_DIR_NAME, name)
    if not os.path.exists(fp):
        os.makedirs( fp, exist_ok=True  )
    return fp

def getTrainingLogDir(name, timestamp=None):
    fp = getTrainingLogRoot(name)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
    fp = fp + "{}/".format(timestamp)

    if not os.path.exists(fp):
        os.mkdir( fp )
    return fp



