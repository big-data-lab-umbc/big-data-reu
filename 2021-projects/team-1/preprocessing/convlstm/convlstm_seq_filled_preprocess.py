# import packages
import pandas as pd
import numpy as np
import tensorflow as tf

'''
This file takes sequential image data and applies a rolling window to the dataset to make it stateless.
The input data begins in shapes (384 months, 448x304 pixels, 11 features) for training data
and (96 months, 448x304 pixels, 11 features) for testing data.

The output data begins in shapes (384 months, 448x304 pixels, 1 feature) for training data
and (96 months, 448x304 pixels, 1 feature) for testing data

A rolling window is then applied to the input data, reshaping it into size (372 samples, 12 months each, 448x304 pixels, and 11 features) for training data
and size (96 samples, 12 months, 448x304 pixels, 1 feature) for testing data

Sample 1 of our input data contains months 1-12, sample 2 contains months 2-13, ..., and sample 372 contains months 372-384.

Output samples stay the same size, with one sample representing the prediction made for the month following the 12 months of input data in each sample  
'''

# import data from numpy
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/x_train_comparison_filled.npy", "rb") as f:
        X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/y_train_comparison_filled.npy", "rb") as f:
        y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/x_test_comparison_filled.npy", "rb") as f:
        X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/y_test_comparison_filled.npy", "rb") as f:
        y_test = np.load(f)

# print data shapes
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Reshaping data to have 4 dimensions
X_train_convlstm_filled_seq = X_train.reshape(-1, 448, 304, 10)
X_test_convlstm_filled_seq = X_test.reshape(-1, 448, 304, 10)
y_train_convlstm_filled_seq = y_train.reshape(-1, 448, 304, 1)
y_test_convlstm_filled_seq = y_test.reshape(-1, 448, 304, 1)

# print new data shapes
print(X_train_convlstm_filled_seq.shape, y_train_convlstm_filled_seq.shape, X_test_convlstm_filled_seq.shape, y_test_convlstm_filled_seq.shape)

# create sliding sequence for every 12 months to convert data into 5 dimensions
# sliding window code from : https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy
def sliding_window(a, window):
	shape = (a.shape[0] - window + 1, window) + a.shape[1:]
	strides = (a.strides[0], ) + a.strides
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#define amount of timesteps
timesteps = 12

# apply function to X train and test sets
X_train_convlstm_filled_seq = sliding_window(X_train_convlstm_filled_seq, window=timesteps)
X_test_convlstm_filled_seq = sliding_window(X_test_convlstm_filled_seq, window=timesteps)

#check data shape
print(X_train_convlstm_filled_seq.shape, X_test_convlstm_filled_seq.shape)
print(X_train_convlstm_filled_seq[0, :, :, :, :])

# Saving final arrays
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/x_train_convlstm_filled_seq_final.npy", "wb") as f:
  np.save(f, X_train_convlstm_filled_seq)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/y_train_convlstm_filled_seq_final.npy", "wb") as f:
  np.save(f, y_train_convlstm_filled_seq)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/x_test_convlstm_filled_seq_final.npy", "wb") as f:
  np.save(f, X_test_convlstm_filled_seq)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/y_test_convlstm_filled_seq_final.npy", "wb") as f:
  np.save(f, y_test_convlstm_filled_seq)

