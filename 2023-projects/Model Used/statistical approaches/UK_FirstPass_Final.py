from datetime import datetime
import pykrige
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import h5py
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from pykrige import variogram_models, UniversalKriging, OrdinaryKriging
from scipy.interpolate import griddata


"""# Define functions"""

def getKNearest(fromX, fromY, measuringPtX, measuringPtY, k):
  # returns a list of k indices of pts from fromX and fromY (assumed to correspond to x and y coordinates of the same pts) that are the nearest to measuringPtX and measuringPtY
  distances = np.sqrt((fromX - measuringPtX)**2 + (fromY - measuringPtY)**2)
  # Get the indices of the k-nearest points
  k_nearest_indices = np.argsort(distances)[:k]
  return k_nearest_indices

def visualizePointsInBox(allTrackX, allTrackY, xCoordMin, xCoordMax, yCoordMin, yCoordMax):
  fig, ax = plt.subplots()
  #ax.scatter(allSurfX, allSurfY, label='Surface points', s=0.2)
  ax.scatter(allTrackX, allTrackY, label='Track bed points', s=0.2)
  # Plot the batches
  # Vertical lines
  ax.axvline(x=xCoordMin, color="black")
  ax.axvline(x=xCoordMax, color="black")
  ax.axhline(yCoordMin, color="black")
  ax.axhline(yCoordMax, color="black")

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_title('Surface data vs track data: batches visualized')
  #ax.legend(loc = "lower right")

  plt.show()
# Store the direction (degrees) of each vector (surf_vx, surf_vy)
def getVectorDir(x, y):
    # Calculate the angle in radians
    #angle_rad = np.arctan2(np.abs(y), x)
    angle_rad = np.arctan2(y,x)
    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    # Ensure the angle is within the range [0, 360)
    angle_deg = (angle_deg + 360) % 360

    #return abs(angle_deg -90) # Subtract 90 b/c 0 degrees = up in the coordinate system used by pykrige. also it just wants the axis, so we constrain to abs
    return angle_deg -90

# Box settings
SQUAREBOXWIDTH = 50 # m
EXPANDSTEP = 10 # m
SHRINKSTEP = EXPANDSTEP
MAX_NUM_PTS = 80
MIN_NUM_PTS = 15

# Which variogram models to even try in order to find the best fit
METHODS = ['exponential', 'power', 'linear', 'gaussian', 'spherical']

# Pick average of K nearest pts' angles from velocity vectors
ANISOTROPY_ANGLE_METH = "KNN"
K = 1

# Define slice of dataset to run on
START_INDEX = 0
END_SLICE_INDEX = 40

TRACKBED_XIND = 7
TRACKBED_YIND = 8
TRACKBED_TARGETIND = 9

# Display plots of variograms?
PLOT = False
RUN_COLAB = False

VERBOSE_OUTPUT = False

# Extract slice of prediction points array from command line args
import sys

if (len(sys.argv) > 1):
  START_INDEX = int(sys.argv[1])
  END_SLICE_INDEX = int(sys.argv[2])
  print("recieved args: ", START_INDEX, " ", END_SLICE_INDEX)

TRACKBED_XIND = 7
TRACKBED_YIND = 8
TRACKBED_TARGETIND = 9

# Display plots of variograms?
PLOT = False
RUN_COLAB = False


if RUN_COLAB:
  pass
else:
  log_file_name = "output_log_" + "(" + str(START_INDEX) + "," + str(END_SLICE_INDEX) + ")" + ".txt"
  sys.stdout = open(log_file_name, "w")

"""# Import data"""

if RUN_COLAB:
  df_merged = pd.read_csv('/content/drive/MyDrive/Predicted and preprocessed data sets /training_test_combined_vmag_ky_230710.csv')
else:
  df_merged =  pd.read_csv('training_test_combined_vmag_ky_230710.csv')
df_merged.columns

"""# Data Preprocessing"""

# Remove duplicates by setting to mean
df2 = df_merged.groupby(['track_bed_x', 'track_bed_y']).agg({'surf_x':'mean', 'surf_y':'mean', 'surf_vx':'mean', 'surf_vy': 'mean', 'surf_elv': 'mean', 'surf_dhdt':'mean', 'surf_SMB': 'mean', 'track_bed_x':'first', 'track_bed_y':'first', 'track_bed_target':'mean', 'v_mag':'mean'})
print("num duplicates found: ", len(df_merged) - len(df2))
df_merged = df2.reset_index(drop=True)
df_merged.columns
print("Number of rows after duplicates removed: ", df_merged.shape)

"""# Execute kriging"""

trackDatTmp = np.array(df_merged)
trackDatTmp.shape
trackDat_x = trackDatTmp[:, TRACKBED_XIND]
trackDat_y = trackDatTmp[:, TRACKBED_YIND]
trackDat_angle = getVectorDir(trackDatTmp[:, 2], trackDatTmp[:, 3])
len(trackDat_angle)

trackDat = np.column_stack((trackDatTmp, trackDat_angle))
trackDat.shape

# For each point in tracking point dataset
  # Grab the track bed data points that are within  within box (except for that that point)
  # Print size of points grabbed in order to ensure it is reasonable
  # train kriging based on that dataset

boxWidth = SQUAREBOXWIDTH
outputCols = ['surf_x', 'surf_y', 'surf_vx', 'surf_vy','surf_elv', 'surf_dhdt', 'surf_SMB', 'track_bed_x', 'track_bed_y', 'v_mag', 'track_bed_target', 'residual', 'prediction', 'estPredictionError']
out = np.empty( (0, 14) ) # (track_bed_x, track_bed_y, target_prediction, target_actual, residual)

slice = df_merged[START_INDEX:END_SLICE_INDEX].copy()
print("\t".join(str(value) for value in outputCols))
Kriging_startime = datetime.now()
for index, row in slice.iterrows():
  #print("Pixel " + str(index + 1) + " out of " + str(df_merged.shape[0]) )
  # Hold the prediction and ground truth for track_bed_target at the current pixel
  predictionThisRow = None
  groundTruthThisRow = row['track_bed_target']

  trackRowsToTrainOn = None

  # Define a spatial batch centered on the point to make the prediction on
  xMin, xMax, yMin, yMax = (None, None, None, None)
  expandStep = EXPANDSTEP
  shrinkStep = SHRINKSTEP
  previousProblem = 0 # 1 = too big; -1 = too small
  while True:
    #print("Top of loop")

    xMin, xMax = (row['track_bed_x'] - boxWidth, row['track_bed_x'] + boxWidth)
    yMin, yMax = (row['track_bed_y'] - boxWidth, row['track_bed_y'] + boxWidth)

    # We are still fiting kriging on the entire dataset (not just the slice)
    x_selected_ind = np.where((trackDat_x >= xMin) & (trackDat_x <= xMax) )[0]
    y_selected_ind = np.where((trackDat_y >= yMin) & (trackDat_y <= yMax) )[0]
    #visualizePointsInBox(trackDat_x, trackDat_y, xMin, xMax, yMin, yMax)

    # Select indices that are in selected x and y range (inside the rectangle) to init object on
    indicesXY = list(set(x_selected_ind).intersection(y_selected_ind))
    trackRowsToTrainOn = np.array([trackDat[ind] for ind in indicesXY])

    # If none were selected, perform same code as if some but too few were selected
    if (len(trackRowsToTrainOn) < MIN_NUM_PTS):

      # Prevent overshooting by reducing step size
      if (previousProblem == 1): # Was too big last time
        expandStep /= 2
      else:
        pass#expandStep *= 1.5

      boxWidth += expandStep
      previousProblem = -1
      continue

    # Remove the row corresponding to the center point
    mask = np.logical_and(trackRowsToTrainOn[:, TRACKBED_XIND] == row['track_bed_x'], trackRowsToTrainOn[:, TRACKBED_YIND] == row['track_bed_y'])
    trackRowsToTrainOn = trackRowsToTrainOn[~mask] # Grab track rows that exclude it
    #print(trackRowsToTrainOn)

    # Print number of points
    numPts = len(trackRowsToTrainOn)
    #print(numPts)

    # Perform iterative resizing of box to ensure correct length
    if (numPts < MIN_NUM_PTS):

      # Prevent overshooting by reducing step size
      if (previousProblem == 1): # Was too big last time
        expandStep /= 2
      else:
        pass#expandStep *= 1.5

      boxWidth += expandStep
      previousProblem = -1
    elif (numPts > MAX_NUM_PTS):

      # Prevent overshooting by reducing step size
      if (previousProblem == -1): # Was too small last time
        shrinkStep /= 2
      else:
        pass#shrinkStep *= 1.5

      boxWidth -= shrinkStep
      previousProblem = 1
    else:
      break # We have chosen an appropriate number of trackRowsToTrainOn
  
  # Define settings for UK covariance model fitting
  nlags = len(trackRowsToTrainOn[:, 1])
  kNearest_PtsToTrainOn_indices = getKNearest(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], row['track_bed_x'], row['track_bed_y'], K)
  angles = trackDat_angle[kNearest_PtsToTrainOn_indices]
  anisotropy_angle = angles.mean()

  # Fit each possible variogram model
  linear = UniversalKriging(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], trackRowsToTrainOn[:, TRACKBED_TARGETIND], variogram_model="linear", anisotropy_scaling = 3, anisotropy_angle = anisotropy_angle, enable_plotting=PLOT, nlags=nlags)
  exponential = UniversalKriging(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], trackRowsToTrainOn[:, TRACKBED_TARGETIND], variogram_model="exponential",anisotropy_scaling = 3, anisotropy_angle = anisotropy_angle, enable_plotting=PLOT, nlags=nlags)
  power = UniversalKriging(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], trackRowsToTrainOn[:, TRACKBED_TARGETIND], variogram_model="power", anisotropy_scaling = 3, anisotropy_angle = anisotropy_angle, enable_plotting=PLOT, nlags=nlags)
  gaussian = UniversalKriging(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], trackRowsToTrainOn[:, TRACKBED_TARGETIND], variogram_model="gaussian",anisotropy_scaling = 3, anisotropy_angle = anisotropy_angle, enable_plotting=PLOT, nlags=nlags)
  spherical = UniversalKriging(trackRowsToTrainOn[:, TRACKBED_XIND], trackRowsToTrainOn[:, TRACKBED_YIND], trackRowsToTrainOn[:, TRACKBED_TARGETIND], variogram_model="spherical",anisotropy_scaling = 3, anisotropy_angle = anisotropy_angle, enable_plotting=PLOT, nlags=nlags)

  model_names = ['linear', 'exponential', 'power', 'gaussian', 'spherical']
  models = [linear, exponential, power,  gaussian, spherical]
  stats = []
  for i, model in enumerate(models):
    stats.append(model.get_statistics()[2])

  # Choose the best model as the one with the lowest capacity ratio
  ind_closest_to_one = np.abs(np.array(stats)).argmin()
  kriging = models[ind_closest_to_one]
  
  # Predict the ground truth pixel
  prediction, predictionVariance = kriging.execute("points", row['track_bed_x'], row['track_bed_y'])
  predictionThisRow = prediction[0]

  residualThisRow = groundTruthThisRow - predictionThisRow
  existingRowData = row[['surf_x', 'surf_y', 'surf_vx', 'surf_vy','surf_elv', 'surf_dhdt', 'surf_SMB', 'track_bed_x', 'track_bed_y', 'v_mag', 'track_bed_target']].copy() # Is this a copy...?
  existingRowData['prediction'] = predictionThisRow  
  existingRowData['residual'] = row['track_bed_target'] - existingRowData['prediction']
  existingRowData['estPredictionError'] = predictionVariance
  #print(existingRowData)
  print("\t".join(str(value) for value in existingRowData.values.tolist()))
  out = np.vstack( (out, np.array(existingRowData)))

Kriging_stoptime = datetime.now()
print("---------------------------------------------------------")
df_krigout = pd.DataFrame(columns = outputCols, data=out)
df_krigout.to_csv( "./pixels (" + str(START_INDEX) + "," + str(END_SLICE_INDEX) + ") residuals.csv" )
sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Finished applying kriging to pixels: {START_INDEX}:{END_SLICE_INDEX}")
print("Time to run: ", Kriging_stoptime - Kriging_startime)