import pandas as pd
import numpy as np

# Load .npz data files containing all variables
data = np.load('/umbc/xfs1/cybertrn/sea-ice-prediction/data/merged_data/1979_2019_combined_data_25km.npz')
new_data = np.load('/umbc/xfs1/cybertrn/sea-ice-prediction/data/merged_data/2020_2021_combined_data_25km.npz')

for key in data.keys():
	print(key)

# Create lists of days per month
months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
leap_months = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_lst = []

# Create a full list of days per month for each month in 1979-2019
for i in range(1979, 2020):
   if i % 4 == 0:
        month_lst.append(leap_months)
   else:
        month_lst.append(months)
month_lst = np.array(month_lst).reshape(-1)

# List of days per month for January 2020-June 2021
new_month_lst = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30]

# Iterate through each variable add to the final array. 
final_arr = np.zeros((len(month_lst) + len(new_month_lst), 448, 304, 10))
var_idx = 0
for key in data.keys():
	print(key)
	if key == 'time' or key == 'lat' or key == 'lon':
		continue
	day_sum = 0
	var = data[key]
	var = np.where(var == -999.0, float("NaN"), var) # Specify missing values as NaN
	new_var = new_data[key]
	new_var = np.where(new_var == -999.0, float("NaN"), var) # Specify missing values as NaN
	if key == 'sea_ice': # Any sea ice value above 252 indicates land or missing. Convert these values to NaN.
		new_var = np.where(new_var > 252.0, float("NaN"), new_var)
	var_arr = np.zeros((len(month_lst) + len(new_month_lst), 448, 304))
	print(var.shape)
	print(var_arr.shape)
	for i in range(len(month_lst)): # Calculate monthly means of the variable from 2020-2021 and add to overall array
		var_arr[i, :, :] = np.nanmean(var[day_sum:(day_sum + month_lst[i]), :, :], axis=0)
		if i % 12 == 0:
			print(day_sum)
		day_sum = day_sum + month_lst[i]
	day_sum = 0
	for j in range(len(new_month_lst)): # Calculate montly means of the variable from 2020-2021 and add to overall array
		var_arr[len(month_lst) + j, :, :] = np.nanmean(new_var[day_sum:(day_sum + new_month_lst[j]), :, :], axis=0)
		if j % 12 == 0:
			print(day_sum)
		day_sum = day_sum + new_month_lst[j]
	final_arr[:, :, :, var_idx] = var_arr # Add variable monthly means to final array.
	var_idx += 1
print(final_arr.shape)
print(final_arr[:, :, :, -1])

# Save final data array with shape (510, 448, 304, 10) to a numpy array.
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/whole_data.npy", "wb") as f:
	np.save(f, final_arr)
