import pykrige
from pykrige import variogram_models, UniversalKriging, OrdinaryKriging
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_combs_from_xy(x_ticks, y_ticks):
  x_all_comb = np.tile(x_ticks, len(y_ticks))
  y_all_comb = np.repeat(y_ticks, len(x_ticks))
  return x_all_comb, y_all_comb

def get_flat_mapping(v):
  v_flat = v.flatten()
  return v_flat

def plotHeatMap(target_var_values, x_values, y_values, title):
  x = x_values
  y = y_values
  X, Y = np.meshgrid(x_values,y_values)

  plt.xticks(rotation=90)

  plt.imshow(target_var_values, cmap='hot', origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
  plt.colorbar()  # Add colorbar
  plt.xlabel('x')  # x-axis label
  plt.ylabel('y')  # y-axis label
  plt.title(title)  # Plot title
  plt.show()  # Display the plot


h5_file_location = 'hackathon.h5'
df = h5py.File(h5_file_location, 'r')

df_predictPts = pd.read_csv("trackCombined.csv")


x = get_flat_mapping(np.asarray(df.get('surf_x')))
y = get_flat_mapping(np.asarray(df.get('surf_y')))
print(x.shape)
print(y.shape)

surf_vx_arr = get_flat_mapping(np.asarray(df.get('surf_vx')))
surf_vy_arr = get_flat_mapping(np.asarray(df.get('surf_vy')))
surf_dhdt_arr = get_flat_mapping(np.asarray(df.get('surf_dhdt')))
surf_SMB_arr = get_flat_mapping(np.asarray(df.get('surf_SMB')))
surf_elv_arr = get_flat_mapping(np.asarray(df.get('surf_elv')))

# Split by each variable
vars = [surf_vx_arr, surf_vy_arr, surf_dhdt_arr, surf_SMB_arr, surf_elv_arr]
varNames = ["surf_vx", "surf_vy", "surf_dhdt", "surf_SMB", "surf_elv"]

# Get which feature we are interpolating from command line args
curVarName = sys.argv[1]
CURVARIND = varNames.index(curVarName)
voi_flat = get_flat_mapping(vars[CURVARIND])
print("Running for ", varNames[CURVARIND])

# Define output grid...x and y ticks of pts to predict
track_bed_x = df_predictPts['track_bed_x']
track_bed_y = df_predictPts['track_bed_y']
track_bed_target = df_predictPts['track_bed_target']


print(x.shape)
print(y.shape)
cond_pos = np.hstack((x.reshape(x.shape[0], 1),y.reshape(x.shape[0], 1)))
print(len(cond_pos)) # (2, 1201)
print(len(vars[0]))
cond_pos_r = np.reshape(cond_pos, (x.shape[0], 2))
print(len(cond_pos_r))
print(cond_pos[0])
print(x[0], ",", y[0])
print(cond_pos[1])
print(x[1], ",", y[1])


# Define function to split ground truth points (pp_x, pp_y) and surface points (x,y) into spatial batches
# a x b batches
def split_into_batches(pp_x, pp_y, x, y, a, b): # returns the indices to split for each batch -> BatchManager

  # Calculate x and y ranges using grid (since grid is slightly wider/taller than xy bounds)
  x_range = {"min":pp_x.min() , "max": pp_x.max(), "range":pp_x.max()-pp_x.min()} # use bc grid data is wider
  y_range = {"min":pp_y.min() , "max": pp_y.max(), "range":pp_y.max()-y.min()}
  x_batch_size = x_range["range"]/a
  y_batch_size = y_range["range"]/b

  print("batch range x: ", x_batch_size)
  print("batch range y: ", x_batch_size)

  #easyPlot(pp_x, pp_y, x, y, x_range, y_range, x_batch_size, y_batch_size, a, b, "batches")
  # Split x and gridx by a
  x_ind_ranges = []
  pp_x_ind_ranges = []

  Batches = np.empty( (b, a) , dtype=object) # b x a matrix

  for i in range(b):
    y_bounds_meters = {"min":y_range["min"] + y_batch_size*i, "max":y_range["min"] + y_batch_size*(i+1) }
    for j in range(a):
      out_dict = {} # {'xy': [list of indices to slice], 'predictxy': [list of indices to slice]}

      # For xy
      x_bounds_meters = {"min":x_range["min"] + x_batch_size*j, "max":x_range["min"] + x_batch_size*(j+1) }
      x_selected_ind = np.where((x >= x_bounds_meters['min']) & (x <= x_bounds_meters['max']) )[0] # x selected ind
      y_selected_ind = np.where((y >= y_bounds_meters['min']) & (y <= y_bounds_meters['max']) )[0]
      xy_indicesToSlice = list(set(x_selected_ind).intersection(y_selected_ind)) # Get slice indices

      # For predictxy
      pp_x_selected_ind = np.where((pp_x >= x_bounds_meters['min']) & (pp_x <= x_bounds_meters['max']) )[0] # x selected ind
      pp_y_selected_ind = np.where((pp_y >= y_bounds_meters['min']) & (pp_y <= y_bounds_meters['max']) )[0]
      predict_xy_indicesToSlice = list(set(pp_x_selected_ind).intersection(pp_y_selected_ind)) # Get slice indices

      out_dict['xy'] = xy_indicesToSlice
      out_dict['pred_pts_xy'] = predict_xy_indicesToSlice

      Batches[i,j] = out_dict
  return Batches

len(vars)


# Define a function to visualize spatial batches
def easyPlot(px, py, x,y, x_range, y_range, x_batch_size, y_batch_size, a, b, title, drawBatches=True):
  fig, ax = plt.subplots()
  ax.scatter(x, y, label='Surface data')
  ax.scatter(px, py, label='Track data (interpolate)', s=0.2)
  # Vertical lines
  for i in range(a+1):
    x_ = x_range["min"] + x_batch_size*i
    ax.axvline(x=x_, color="yellow")

  # Horizontal lines
  for i in range(b+1):
    y_ = y_range["min"] + y_batch_size*i
    ax.axhline(y_, color="yellow")

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title('Surface data vs track data: batches visualized')
  #ax.legend(loc = "lower right")
  plt.show()

# rows of (meanfield, varfield)
output = [] # = np.empty((num_batches, 2)) # (mean_field, var_field)

# var, batch, mean_field, var_field
output_all_vars = [] # np.empty((len(vars), num_batches, 2)) # (var, num_batches, 2
x_fix=x
y_fix=y

# Flatten vals for all vars
allvars_flat = np.empty( (len(x), len(vars)) )
for ind, voi_vals in enumerate(vars):
  allvars_flat[:,ind] = get_flat_mapping(voi_vals)

print(x_fix.shape)
print(allvars_flat.shape)

# Split data
NUM_X_BATCHES = 15
NUM_Y_BATCHES = 5
Batches_M = split_into_batches(track_bed_x, track_bed_y, x_fix, y_fix, NUM_X_BATCHES, NUM_Y_BATCHES) # 15 x 1 matrix of dicts
# dict form: {'xy' : xy_indicesToSlice, 'pred_pts_xy' : predict_xy_indicesToSlice}
# Batches_M is 15 x 1 matrix of dicts
# dict form: {'xy' : xy_indicesToSlice, 'pred_pts_xy' : predict_xy_indicesToSlice}
print(x_fix.min())
print(x_fix.max())
print(y_fix.min())
print(y_fix.max())

import random
# Interpolate each batch
df_out_big = pd.DataFrame()
for i_yBatch in range(NUM_Y_BATCHES):
  for i_xBatch in range(NUM_X_BATCHES):

    df_out = pd.DataFrame(columns=["surf_x", "surf_y"])
    print("Batch ", f"({i_yBatch}, {i_xBatch})")
    # Get current batch indices
    xy_cbatch_indices = Batches_M[i_yBatch][i_xBatch]['xy'] # Yields x indices to slice at
    pred_cbatch_indices = Batches_M[i_yBatch][i_xBatch]['pred_pts_xy']

    # cut down indices from xy. try to get representative sample
    num_pts = int((len(xy_cbatch_indices) - 2) / 30) # keep 1/30th
    rand_sample = random.sample(xy_cbatch_indices[1:-1], num_pts)
    print("r", len(rand_sample))
    xy_cbatch_indices = [xy_cbatch_indices[0]] + rand_sample + [xy_cbatch_indices[-1]]

    print("num xy in batch: ", len(xy_cbatch_indices))
    print("num pred_xy in batch: ", len(pred_cbatch_indices))

    if (len(pred_cbatch_indices) < 1):
      print("skipping empty batch")
      continue

    # Vals at indices
    x_fix_cbatch = np.array([x_fix[ind] for ind in xy_cbatch_indices])
    print(x_fix_cbatch.dtype)
    y_fix_cbatch = np.array([y_fix[ind] for ind in xy_cbatch_indices])

    voi_flat_cbatch = np.array([voi_flat[ind] for ind in xy_cbatch_indices])

    voi_flat_cbatch_list = [voi_flat_cbatch]

    track_bed_x_cbatch = np.array([track_bed_x[ind] for ind in pred_cbatch_indices])
    track_bed_y_cbatch = np.array([track_bed_y[ind] for ind in pred_cbatch_indices])
    track_bed_target_cbatch = np.array([track_bed_target[ind] for ind in pred_cbatch_indices])

    df_out['track_bed_x'] = track_bed_x_cbatch
    df_out['track_bed_y'] = track_bed_y_cbatch
    df_out["track_bed_target"] = track_bed_target_cbatch

    for voi_flat_cbatch in voi_flat_cbatch_list:
      # Choose covariance model

      PLOT = False
      # Fit each of the variogram models...
      nlags = len(voi_flat_cbatch)
      linear = UniversalKriging(x_fix_cbatch,  y_fix_cbatch, voi_flat_cbatch, variogram_model="linear", anisotropy_scaling = 3, anisotropy_angle = 90, enable_plotting=PLOT, nlags=nlags)
      exponential = UniversalKriging(x_fix_cbatch,  y_fix_cbatch, voi_flat_cbatch, variogram_model="exponential", anisotropy_scaling = 3, anisotropy_angle = 90, enable_plotting=PLOT, nlags=nlags)
      power = UniversalKriging(x_fix_cbatch,  y_fix_cbatch, voi_flat_cbatch, variogram_model="power", anisotropy_scaling = 3, anisotropy_angle = 90, enable_plotting=PLOT, nlags=nlags)
      gaussian = UniversalKriging(x_fix_cbatch,  y_fix_cbatch, voi_flat_cbatch, variogram_model="gaussian", anisotropy_scaling = 3, anisotropy_angle = 90, enable_plotting=PLOT, nlags=nlags)
      spherical = UniversalKriging(x_fix_cbatch,  y_fix_cbatch, voi_flat_cbatch, variogram_model="spherical", anisotropy_scaling = 3, anisotropy_angle = 90, enable_plotting=PLOT, nlags=nlags)

      model_names = ['linear', 'exponential', 'power', 'gaussian', 'spherical']
      models = [linear, exponential, power,  gaussian, spherical]

      stats = []
      for i, model in enumerate(models):
        print(model_names[i])
        print(model.get_epsilon_residuals())
        model.print_statistics()
        stats.append(model.get_statistics()[2])
        #stats.append(np.mean(np.abs(model.get_epsilon_residuals())))
        print(np.mean(np.abs(model.get_epsilon_residuals())))

      #stats = [linear.get_statistics()[1], exponential.get_statistics()[1], power.get_statistics()[1], gaussian.get_statistics()[1], spherical.get_statistics()[1]]
      print(stats)

      # Get model with the smallest capacity ratio (best error)
      ind_closest_to_one = np.abs(np.array(stats)).argmin()
      kriging = models[ind_closest_to_one]
      print("Best variogram: ", model_names[ind_closest_to_one])

      print("Making prediction for current pixel...")
      mean_field, var_field = kriging.execute("points", track_bed_x_cbatch, track_bed_y_cbatch)

      df_out[varNames[CURVARIND]] = mean_field
      print("finished for: ", varNames[CURVARIND])

    # Concat to master df
    df_out_big = pd.concat([df_out_big, df_out])

print("All finished!")

# Save off csv
df_out_big.to_csv(f"{varNames[CURVARIND]}.csv")
