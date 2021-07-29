#TODO: Parameterize variables (lag, test size (in months), len(data))
import pandas as pd
import numpy as np

# Open full data array with shape (510, 448, 304, 10) corresponding to (months, height, width, # of predictors)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/whole_data.npy", "rb") as f:
	data = np.load(f)

# Select the SIC variable
ice = data[:, :, :, -1]

# Adding North Pole Hole to land mask
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_actual.npy", "rb") as f:
	orig_mask = np.load(f)

orig_mask = np.tile(orig_mask, (510, 1, 1))
orig_mask[:, 208:260, 120:180] = ~np.isnan(ice[:, 208:260, 120:180])

# Save land mask with shape (510, 448, 304) with a separate land mask for each month.
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_final_whole.npy", "wb") as f:
	np.save(f, orig_mask)

# Fill North Pole Hole in the SIC data
ice[:, 208:260, 120:180][np.isnan(ice[:, 208:260, 120:180])] = 100.0

data[:, :, :, -1] = ice

# Train-Test Split. This code is for a lag of two, training on 1979-2021, and testing on 2013-2020.
X_train = data[:406, :, :, :]
X_test = data[406:-8, :, :, :]
y_train = data[2:408, :, :, -1]
y_test = data[408:-6, :, :, -1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Save train and test datasets to numpy arrays.
with open("x_train_whole_lag_two.npy", "wb") as f:
	np.save(f, X_train)
with open("y_train_whole_lag_two.npy", "wb") as f:
	np.save(f, y_train)
with open("x_test_whole_lag_two.npy", "wb") as f:
	np.save(f, X_test)
with open("y_test_whole_lag_two.npy", "wb") as f:
	np.save(f, y_test)


