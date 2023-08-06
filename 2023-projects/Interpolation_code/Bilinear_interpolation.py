#Interpolation of ICE Bed Training Data using scipy.interpolate.RegularGridInterpolator method

#2D Interpolation of all 5 variables.

#Src: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator


import h5py
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

h5_file_location = '/scratch1/09008/halam3/hackathon.h5'
df = h5py.File(h5_file_location, 'r')

track_bed_training =df.get('track_bed_training')
track_bed_training = pd.DataFrame(track_bed_training)
track_bed_testing =df.get('track_bed_testing')
track_bed_testing = pd.DataFrame(track_bed_testing)
surf_x =df.get('surf_x')
surf_x = pd.DataFrame(surf_x)
surf_y =df.get('surf_y')
surf_y = pd.DataFrame(surf_y)
surf_SMB=df.get('surf_SMB')
surf_SMB = pd.DataFrame(surf_SMB)
surf_dhdt=df.get('surf_dhdt')
surf_dhdt = pd.DataFrame(surf_dhdt)
surf_elv=df.get('surf_elv')
surf_elv = pd.DataFrame(surf_elv)
surf_vx=df.get('surf_vx')
surf_vx = pd.DataFrame(surf_vx)
surf_vy=df.get('surf_vy')
surf_vy = pd.DataFrame(surf_vy)

# converting dataframe to numpy array
track_bed_training_T = track_bed_training.T
track_bed_training_T_arr = track_bed_training_T.to_numpy()

track_bed_testing_T = track_bed_testing.T
track_bed_testing_T_arr = track_bed_testing_T.to_numpy()

surf_SMB_arr = surf_SMB.to_numpy()
surf_dhdt_arr = surf_dhdt.to_numpy()
surf_elv_arr = surf_elv.to_numpy()
surf_vx_arr = surf_vx.to_numpy()
surf_vy_arr = surf_vy.to_numpy()

#checking column sequence in test dataset of size 1201x1201
data_1201=pd.read_csv('/scratch1/09008/halam3/df_1201.csv')
list(data_1201.columns.values) #list(test_data_1201.columns.values)

# concatenating all 5 variables into one data array
data_all = np.zeros((5,1201,1201))

# 'surf_vx', 'surf_vy','surf_dhdt','surf_SMB', 'surf_elv'
data_all[0,:,:]= surf_vx_arr
data_all[1,:,:]= surf_vy_arr
data_all[2,:,:]= surf_dhdt_arr
data_all[3,:,:]= surf_SMB_arr
data_all[4,:,:]= surf_elv_arr

data_all_T = data_all.transpose(1, 2, 0)  # tranforming from (5, 1201, 1201) to (1201, 1201, 5)

data_all_T.shape

data_all_T[1,1,0], surf_vx_arr[1,1]

# numpy array to store the training data with interpolated variables

track_bed_training_T_interpotale = np.zeros((track_bed_training_T_arr.shape[0],8))
track_bed_training_T_interpotale[:,0:3] = track_bed_training_T_arr

track_bed_training_T_interpotale[0,:], track_bed_training_T.iloc[0], track_bed_training_T_arr[0,:]

# interpolation of training data using surf_x(0) and surf_y(1)

from scipy.interpolate import RegularGridInterpolatorx
import numpy as np
import matplotlib.pyplot as plt

x = surf_x.iloc[:][0]
y = surf_y.iloc[0][:]

interp = RegularGridInterpolator((x, y), data_all_T, bounds_error=False, fill_value=None)

for i in range(track_bed_training_T_interpotale.shape[0]):
  int_data = interp((track_bed_training_T_arr[i,0], track_bed_training_T_arr[i,1]))
  track_bed_training_T_interpotale[i,3:] = int_data  
  print(i)

# convert interpolated data from numpy to dataframe 

column_values = ['surf_x', 'surf_y', 'track_bed_target', 'surf_vx', 'surf_vy','surf_dhdt','surf_SMB', 'surf_elv']
track_bed_training_T_interpotale_df = pd.DataFrame(data = track_bed_training_T_interpotale, columns = column_values)
track_bed_training_T_interpotale_df

## Writing interpolated data in CSV format

track_bed_training_T_interpotale_df.to_csv('/scratch1/09008/halam3/track_bed_training_interpotale.csv')
