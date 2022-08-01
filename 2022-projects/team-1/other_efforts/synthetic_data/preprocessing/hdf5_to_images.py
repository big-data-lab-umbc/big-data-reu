from matplotlib import pyplot as plt
import h5py
import numpy as np
from PIL import Image
import os
import sys
os.chdir( ".." )
sys.path.append('./')

from skimage.transform import rescale

from utils.paths import DATASET_PATHS, HDF5_ROOT
from sys import argv
target_size = int(argv[1])

do_fft = True
if len(argv) == 3:
    if argv[2] == "nofft":
        do_fft = False
    else:
        raise Exception( "The only value we actually take for the second argument is nofft" )

target_dataset = "{}data_{}".format( "fft_" if do_fft else "", target_size )

# where should we put the created pngs
try:
    dest = DATASET_PATHS[target_dataset]
except:
    raise Exception( "The path to {} needs to be defined in utils.paths.DATASET_PATHS".format(target_dataset) )

if not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)

def fftFilter(arr):
    from scipy import fftpack
    fft = fftpack.fft2(arr)
    keep_fraction = 0.12

    r, c = fft.shape

    fft[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

    fft[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

    arrp = fftpack.ifft2(fft).real
    return arrp

def convertArray(arr):
    # normalize median to 0.5
    arr = (arr - np.min(arr)) 
    arr = arr * 0.5 / np.median(arr)
    arr = arr.clip(0,1)

    # transform the approximately normal distr over intensity to approximately uniform
    from scipy.stats import norm
    (mu, sigma) = norm.fit(arr)

    cdf_v = norm.cdf(arr, loc=mu, scale=sigma)
    arr = cdf_v.clip(0,1)

    return arr

def resizeArray(arr, target_size):
    return rescale(arr, target_size / 1000., anti_aliasing=True, preserve_range=True)
    

def doTheThing(fn):
    fnp = fn[:-4] + "png"
    
    hdf = h5py.File(HDF5_ROOT + fn,'r')
    
    arr = hdf["DNB_observations"][:]
    arr = convertArray(arr)

    if do_fft:
        arr = fftFilter(arr)

    arr = resizeArray( arr, target_size )

    img = (arr * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(dest + fnp)

assert os.path.exists(dest)
for path, dirs, files in os.walk(HDF5_ROOT):
    for f in files:
        if f[-4:] != "hdf5": continue
        doTheThing(f)