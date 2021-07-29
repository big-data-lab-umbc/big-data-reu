'''
Author: Eliot Kim

Purpose: Build and train a CNN model which outputs a 448 by 304 grid of predicted sea ice concentrations for each month.

Data: One-month lag and filled North Pole Hole
Train: 1979-2012
Test: 2013-2020
Loss Function: Land Mask
'''
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/x_train_whole_lag_one.npy", "rb") as f:
  X_train = np.load(f)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_train_whole_lag_one.npy", "rb") as f:
  y_train = np.load(f)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/x_test_whole_lag_one.npy", "rb") as f:
  X_test = np.load(f)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_test_whole_lag_one.npy", "rb") as f:
  y_test = np.load(f)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_actual.npy", "rb") as f:
  y_land_mask = np.load(f)

# Change all remaining NaN values to 0's.
# TODO: Deal with variables other than sea ice with NaN values --> SST, etc.
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

tf.random.set_seed(123) # Set random seed for reproducible results.

# Custom loss function to mask out land pixels from predicted concentrations
def custom_rmse(y_true, y_pred):
	y_pred = tf.math.multiply(y_pred, y_land_mask)
	squared_difference = tf.square(y_true - y_pred)
	means = tf.reduce_mean(squared_difference)
	return means

# Model Building
def create_cnn():
	cnn = keras.models.Sequential()
	cnn.add(keras.layers.Conv2D(128, (5,5), input_shape = (448, 304, 10)))
	cnn.add(keras.layers.MaxPooling2D((2,2)))
	cnn.add(keras.layers.Conv2D(32, (5,5), activation="relu"))
	cnn.add(keras.layers.MaxPooling2D((2,2)))
	cnn.add(keras.layers.Conv2D(8, (5,5), activation="relu"))
	cnn.add(keras.layers.Flatten())
	cnn.add(keras.layers.Dense(256, activation="relu"))
	cnn.add(keras.layers.Dense(448*304, activation="linear"))
	cnn.add(keras.layers.Reshape((448, 304), input_shape = (448*304,)))
	
	cnn.compile(optimizer="adamax", loss=custom_rmse, metrics=[tf.keras.metrics.MeanSquaredError()], run_eagerly=True)
	return cnn

# Model Initialization
cnn = create_cnn()

# Display model info
print(cnn.summary())

# Model Training
history = cnn.fit(X_train, y_train, epochs=400, batch_size=4, validation_data=(X_test, y_test), callbacks=keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True))

# Calculate train and test metrics
train_loss, train_mse = cnn.evaluate(X_train, y_train)
print("Train rMSE: {:.4f}\nTest Loss: {:.4f}".format(math.sqrt(train_mse), train_loss))

test_loss, test_mse = cnn.evaluate(X_test, y_test)
print("Test rMSE: {:.4f}\nTest Loss: {:.4f}".format(math.sqrt(test_mse), test_loss))

# Post-process predicted SIC
pred_ice = cnn.predict(X_test)
real_ice = y_test

with open("/home/ekim33/reu2021_team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
	land_mask = np.load(f)

# Multiply predicted ice values by the monthly land mask, set values below 0 to 0, and set values above 100 to 100. 
land_mask = land_mask[408:-6]
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
print("Post-Processed MSE: ", mse, "\n", "RMSE: ", rmse, "\n")

# Save real and post-processed predicted SIC values to numpy arrays.
with open("/home/ekim33/reu2021_team1/research/evaluation/pred_ice_comparison_base_cnn_maskthresh_lag_one_small_batch.npy", "wb") as f:
	np.save(f, pred_ice)
with open("/home/ekim33/reu2021_team1/research/evaluation/real_ice_comparison_base_cnn_maskthresh_lag_one_small_batch.npy", "wb") as f:
	np.save(f, real_ice)


