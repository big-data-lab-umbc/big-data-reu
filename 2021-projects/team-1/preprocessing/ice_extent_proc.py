'''
Purpose: Preprocess ice extent data into a list of 510 monthly values. Plot predicted vs real ice extents.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Preprocess ice extent data to a time range of 1979-2021 and fill missing values with simple local averaging.
path = "/umbc/xfs1/cybertrn/sea-ice-prediction/data/arctic_data_csv_monthly/monthly_seaice_extent/"
file_list = os.listdir(path)
l = []

# Parse through each month's file and add relevant data to the list of extent values. 
for i in range (1, 13):
	for f in file_list:
		if f[2:4] == str(i).zfill(2):
			file_name = f
			print(path + file_name)
			df = pd.read_csv(path + f)
			print(df.head())
			print(df.tail())
			l.append(df[' extent'][(df['year'] > 1978) & (df['year'] < 2021)])

# Convert list of extent values to a pandas DataFrame. 
df = pd.DataFrame(l).transpose()
df.iloc[:, -1] = df.iloc[:, -1].shift(-1)
df.iloc[:, -2] = df.iloc[:, -2].shift(-1)
df.drop(df.tail(1).index, inplace=True)
df_flat = df.to_numpy().flatten()

# Fill the two missing values with average of two nearest values.
df_flat[107] = (df_flat[106] + df_flat[109]) / 2
df_flat[108] = (df_flat[106] + df_flat[109]) / 2

# Add 2021 extent values to DataFrame.
for i in range (1, 13):
	for f in file_list:
		if f[2:4] == str(i).zfill(2):
			file_name = f
			print(path + file_name)
			df_temp = pd.read_csv(path + f)
			print(df.head())
			print(df.tail())
			df_flat = np.append(df_flat, df_temp[' extent'][(df_temp['year'] == 2021)])

# Save extent values to a numpy array.
with open("real_ice_extents.npy", "wb") as f:
	np.save(f, df_flat)

# Plot ice extent model results. Example provided is for the Multi-Task CNN model predictions. Time range: 2013-2020.
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/evaluation/pred_extent_multiout_cnn_lag_one.npy", "rb") as f:
	pred_area = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/evaluation/real_extent_multiout_cnn_lag_one.npy", "rb") as f:
	real_area = np.load(f)
with open("real_ice_extents.npy", "rb") as f:
	actual_area = np.load(f)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 12))
ax.set_title("Predicted vs Actual Arctic Sea Ice Extent, 2013-2020", fontsize=20)

df_pred_area = pd.DataFrame(pred_area)
df_real_area = pd.DataFrame(real_area)
df_actual_area = pd.DataFrame(actual_area[408:504])

df_pred_area.plot(ax=ax)
df_real_area.plot(ax=ax)
df_actual_area.plot(ax=ax)
ax.set_xlabel("Month", fontsize=16)
ax.set_ylabel("Sea Ice Extent (million km^2)", fontsize=16)
ax.set_xticks(range(0, 96, 6))
ax.set_yticks(range(0, 16, 2))
ax.legend(["Extent Loss Multi-Task CNN Predicted", "Real Extent (CSV)"])
plt.margins(x=0)
fig.savefig("ice_extent_areas_multiout_cnn_2013-2020.png")
