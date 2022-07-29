## paths are relative to the parent directory of this one
USER_LOGS = "../../logs/"
# where are the hdf5 files
HDF5_ROOT = "/umbc/xfs1/cybertrn/users/tchapma1/research/hdf5/vnb_patch_data_2018/"

DATASET_PATHS = {
    "data_256": "datasets/data_256/",
    "fft_data_256": "datasets/fft_data_256/",
    "synthetic_data_256": "datasets/synthetic_data_256/",
    "2020_1000": "datasets/2020_1000",
}

import os
import datetime

def getDataDirectory( dataset_name ):
    return DATASET_PATHS[dataset_name]

def getTrainingLogRoot(name, dir_name="git_repo"):
    fp = USER_LOGS + "{}/{}/".format(dir_name, name)
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



