'''
Purpose: Post-process predicted SIC from CNN models: Remove values over land pixels and threshold SIC values to [0, 100] range.
'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import math

# Data Loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/modeling/pred_ice_comparison_base_cnn.npy", "rb") as f:
	pred_ice = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/modeling/real_ice_comparison_base_cnn.npy", "rb") as f:
	real_ice = np.load(f)
with open("/home/ekim33/reu2021_team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
	land_mask = np.load(f)

# Post-Processing
land_mask = land_mask[408:-6]
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

# Calculate Post-processed MSE and RMSE
mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
print("MSE: ", mse, "\n", "RMSE: ", rmse, "\n")

# Save post-processed SIC images to numpy arrays
with open("/home/ekim33/reu2021_team1/research/evaluation/pred_ice_comparison_base_cnn_maskthresh.npy", "wb") as f:
	np.save(f, pred_ice)
with open("/home/ekim33/reu2021_team1/research/evaluation/real_ice_comparison_base_cnn_maskthresh.npy", "wb") as f:
	np.save(f, real_ice)

