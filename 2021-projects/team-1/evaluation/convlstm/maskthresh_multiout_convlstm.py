import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import math

'''
Evaluate the RMSE of predicted SIC values, returning a numpy file.
Values are post-processed. All SIC values below 0 are converted to 0, and all values above 100 are converted to 100.
RMSE is then evaluated.
'''

#load data
with open("multiout_filled_convlstm_image_rolling_preds.npy", "rb") as f:
	pred_ice = np.load(f)
with open("multiout_filled_convlstm_image_rolling_actual.npy", "rb") as f:
	real_ice = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/modeling/y_land_mask_actual.npy", "rb") as f:
	land_mask = np.load(f)

# reshape data into 3 dimensions
pred_ice = pred_ice.reshape(84, 448, 304)
real_ice = real_ice.reshape(84, 448, 304)
land_mask = land_mask.reshape(448, 304)

#clip values below 0 and greater than 100
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

#calculate and print RMSE
mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
nrmse = rmse / np.mean(real_ice)
print("MSE: ", mse, "\n", "RMSE: ", rmse, "\n", "NRMSE:", nrmse)

# save post-processed values
with open("multiout_convlstm_rolling_pred_ice.npy", "wb") as f:
	np.save(f, pred_ice)
