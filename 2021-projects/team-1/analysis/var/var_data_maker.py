'''
Purpose: Generate train and test data files for the VAR model.
'''
import pandas as pd
import numpy as np

# Open full dataset with shape (510, 448, 304, 10) corresponding to (months, height, width, # of predictors).
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/whole_data.npy", "rb") as f:
	whole_data = np.load(f)

# Convert NaNs to 0.0 and average over the spatial area.
whole_data = np.nan_to_num(whole_data)
whole_data = np.mean(whole_data, axis=(1,2))

# Load ice extent data
extents = np.load("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy")
extents = np.expand_dims(extents, axis=1)

# Train-test split. This code trains on 1979-2012 and tests on 2013-2020.
var_train = whole_data[:408, :]
var_test = whole_data[408:504]

# Add extents to train and test data.
var_train = np.concatenate((var_train, extents[:408]), axis=1)
var_test = np.concatenate((var_test, extents[408:504]), axis=1)

var_train = pd.DataFrame(var_train)
var_test = pd.DataFrame(var_test)

# Save train and test data to csvs. 
var_train.to_csv("var_final_train.csv")
var_test.to_csv("var_final_test.csv")
