import pandas as pd
import numpy as np

# load whole dataset
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/whole_data.npy", "rb") as f:
	data = np.load(f)

# print data
print(data[:, :, :, :])

# remove ice maps from data
ice = data[:, :, :, -1]

# Fill North Pole Hole
ice[:, 208:260, 120:180][np.isnan(ice[:, 208:260, 120:180])] = 100.0

# replace ice maps in data
data[:, :, :, -1] = ice

# Train-Test Split
train_months = 408

# Train-Test Split
X_train = data[:train_months, :, :, :]
X_test = data[train_months:504, :, :, :]
y_train = data[:train_months, :, :, -1]
y_test = data[train_months:504, :, :, -1]

# print dataset shapes
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# save data
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/x_train_comparison_filled.npy", "wb") as f:
	np.save(f, X_train)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/y_train_comparison_filled.npy", "wb") as f:
	np.save(f, y_train)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/x_test_comparison_filled.npy", "wb") as f:
	np.save(f, X_test)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Comparison Data/y_test_comparison_filled.npy", "wb") as f:
	np.save(f, y_test)


