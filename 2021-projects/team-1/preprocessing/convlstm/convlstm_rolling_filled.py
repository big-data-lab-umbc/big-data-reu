import pandas as pd
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.compat.v1.keras import backend as K

'''
Data preprocessing for Convlstm with rolling window.
Loads rolling, sequenced data and removes first 12 months of output observations and last month of training observations.
This allows the network to use the first months 1-12 of input data to predict the output for the 13th month, months 2-13 to predict the output for month 14, etc. 
'''
# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/x_train_convlstm_filled_seq_final.npy", "rb") as f:
        X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/y_train_convlstm_filled_seq_final.npy", "rb") as f:
        y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/x_test_convlstm_filled_seq_final.npy", "rb") as f:
        X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Sequence Data/y_test_convlstm_filled_seq_final.npy", "rb") as f:
        y_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
        y_extent = np.load(f)

#Split to ~80%-20% train-test split
train_months=408
# 408 months (34 years) train - 96 months (8 years) test
y_extent_train=y_extent[:train_months]
y_extent_test=y_extent[train_months:504]

# convert nan values to 0
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

#specify lead_time
lead_time=1

#Remove last lead_time months in train and test data input
X_train = np.array(X_train)[:-lead_time, :, :, :, :]
X_test = np.array(X_test)[:-lead_time, :, :, :, :]

# Remove first 11+lead_time months in train and test output to make input and output lengths equal
y_train = y_train[11+lead_time:, :, :, :]
y_test = y_test[11+lead_time:, :, :, :]
y_extent_train = y_extent_train[11+lead_time:]
y_extent_test = y_extent_test[11+lead_time:]

# print data shape:
print("X_train shape:{} \nX_test shape:{} \ny_train shape: {} \ny_test shape: {} \ny_extent_train_shape: {} \ny_extent_test_shape: {}"
	.format(X_train.shape, X_test.shape, y_train.shape, y_test.shape, y_extent_train.shape, y_extent_test.shape))

# save files
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_train_rolling_filled_final.npy", "wb") as f:
        np.save(f, X_train)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_test_rolling_filled_final.npy", "wb") as f:
	np.save(f, X_test)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_train_rolling_filled_final.npy", "wb") as f:
	np.save(f, y_train)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_test_rolling_filled_final.npy", "wb") as f:
	np.save(f, y_test)
with open("y_extent_train_rolling_final.npy", "wb") as f:
	np.save(f, y_extent_train)
with open("y_extent_test_rolling_final.npy", "wb") as f:
	np.save(f, y_extent_test)
