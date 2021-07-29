'''
Purpose: Calculate SIE based on SIC images using two different methods.
'''
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math

# Load post-processed predicted and real SIC data
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/evaluation/pred_ice_extent_maskthresh_lag_one_small_batch.npy", "rb") as f:
	ice = np.squeeze(np.load(f))
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/evaluation/real_ice_extent_maskthresh_lag_one_small_batch.npy", "rb") as f:
	real_ice = np.squeeze(np.load(f))
with open("/home/ekim33/reu2021_team1/research/analysis/data/lats.npy", "rb") as f:
	lats = np.load(f)
with open("/home/ekim33/reu2021_team1/research/analysis/data/lons.npy", "rb") as f:
	lons = np.load(f)

# Fill North Pole HOle
ice[:, 208:260, 120:180][np.isnan(ice[:, 208:260, 120:180])] = 100.0
real_ice[:, 208:260, 120:180][np.isnan(real_ice[:, 208:260, 120:180])] = 100.0

# Simple Method
tot_ice_extent = np.sum(ice > 15.0, axis=(1, 2))*625 / 1e6
real_tot_ice_extent = np.sum(real_ice > 15.0, axis=(1,2))*625 / 1e6

# Per-pixel area calculation
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/modeling/data/area_size.npy", "rb") as f:
	areas = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
	actual = np.load(f)
actual = actual[408:504]
areas_extent = np.sum(np.multiply(ice > 15.0, areas), axis=(1,2)) / 1e6
real_areas_extent = np.sum(np.multiply(real_ice > 15.0, areas), axis=(1,2)) / 1e6
print(f"RMSE: {math.sqrt(mean_squared_error(areas_extent, real_areas_extent))}")
print(f"Real Ice RMSE: {math.sqrt(mean_squared_error(areas_extent, actual))}")

# Save calculated SIE values to numpy arrays
with open("area_extents_extent_cnn_lag_one.npy", "wb") as f:
	np.save(f, areas_extent)
with open("real_area_extents_extent_cnn_lag_one.npy", "wb") as f:
	np.save(f, real_areas_extent)
