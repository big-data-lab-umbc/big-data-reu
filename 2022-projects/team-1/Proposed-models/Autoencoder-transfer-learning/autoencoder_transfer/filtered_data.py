from load_labeled_data import _getData
from paths import *

# path = 'whatever data path you need to use'
# replace first argument in return statement with necessary path

def getData(no_shuffle=False):
    return _getData(EXPANDED_FFT, normalize=True, augment=True, rgb=False, batch_size=32, shuffle=not no_shuffle)
