'''
Purpose: Build and train a Multi-Task CNN model which outputs a 448 by 304 grid of predicted sea ice concentrations and a sea ice extent prediction for each month.

Data: One-month lag and filled North Pole Hole
Train: 1979-2012
Test: 2013-2020
Loss Functions: Land Mask SIC and SIE MSE
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

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
	extents = np.load(f)

y_extent_train = extents[(408 - X_train.shape[0]):408]
y_extent_test = extents[408:504]

# Change all remaining NaN values to 0's.
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# Custom loss function to mask out land pixels from predicted concentrations
def custom_mse(y_true, y_pred):
	y_pred = tf.math.multiply(y_pred, y_land_mask)
	squared_difference = tf.square(y_true - y_pred)
	means = tf.reduce_mean(squared_difference)
	return means

# Model Building
class MultiOutputCNN():

	# Shared root branch 
	def make_default_hidden_layers(self, inputs, width, height):
		x = keras.layers.Conv2D(128, (5,5), input_shape = (width, height, 10))(inputs)
		x = keras.layers.MaxPooling2D((2,2))(x)
		x = keras.layers.Conv2D(32, (5,5), activation="relu")(x)
		x = keras.layers.MaxPooling2D((2,2))(x)
		x = keras.layers.Conv2D(8, (5,5), activation="relu")(x)
		x = keras.layers.Flatten()(x)
		x = keras.layers.Dense(256, activation="relu")(x)
		return x

	# SIC branch. Outputs SIC image for each month in the batch.
	def build_sic_branch(self, inputs, width, height):
		x = self.make_default_hidden_layers(inputs, width, height)
		x = keras.layers.Dense(512, activation="relu")(x)
		x = keras.layers.Dense(width*height, activation="linear")(x)
		sic_output = keras.layers.Reshape((width, height, 1), input_shape=(width, height,), name="sic_output")(x)
		return sic_output 

	# Extent branch. Outputs vector of extents for each month in the batch.
	def build_extent_branch(self, inputs, width, height):
		x = self.make_default_hidden_layers(inputs, width, height)
		x = keras.layers.Dense(128, activation="relu")(x)
		x = keras.layers.Dense(32, activation="relu")(x)
		x = keras.layers.Dense(8, activation="relu")(x)
		extent_output = keras.layers.Dense(1, activation="linear", name="extent_output")(x)
		return extent_output

	# Creates and compiles Multi-Task CNN model with an SIC branch and an extent branch. 
	def assemble_full_model(self, width, height, features):
		input_shape = (width, height, features)
		inputs = keras.layers.Input(shape=input_shape)

		sic_branch = self.build_sic_branch(inputs, width, height)
		extent_branch = self.build_extent_branch(inputs, width, height)

		model = keras.models.Model(inputs=inputs, outputs=[sic_branch, extent_branch], name="sea_ice_net")

		model.compile(optimizer="adamax", loss={"sic_output": custom_mse, "extent_output": "mse"}, metrics={"sic_output": keras.metrics.RootMeanSquaredError(), "extent_output": keras.metrics.RootMeanSquaredError()})

		return model

# Model Initialization
cnn_multiout = MultiOutputCNN().assemble_full_model(X_train.shape[1], X_train.shape[2], 10)
print(cnn_multiout.summary())

# Model Fitting
history = cnn_multiout.fit(x=X_train, y=[y_train, y_extent_train], epochs=400, batch_size=32, validation_split=0.2, callbacks=keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True))

# Model Evaluation
sic_train_pred, extent_train_pred = cnn_multiout.predict(X_train)
sic_test_pred, extent_test_pred = cnn_multiout.predict(X_test)

# RMSE Calculations
sic_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), sic_train_pred.flatten()))
extent_train_rmse = math.sqrt(mean_squared_error(y_extent_train, extent_train_pred))

sic_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), sic_test_pred.flatten()))
extent_test_rmse = math.sqrt(mean_squared_error(y_extent_test, extent_test_pred))

# nRMSE calculations
sic_train_nrmse = math.sqrt(mean_squared_error(y_train.flatten(), sic_train_pred.flatten())) / np.mean(y_train)
extent_train_nrmse = math.sqrt(mean_squared_error(y_extent_train, extent_train_pred)) / np.mean(y_extent_train)

sic_test_nrmse = math.sqrt(mean_squared_error(y_test.flatten(), sic_test_pred.flatten())) / np.mean(y_test)
extent_test_nrmse = math.sqrt(mean_squared_error(y_extent_test, extent_test_pred)) / np.mean(y_extent_test)

print(f"Training: \n SIC RMSE: {sic_train_rmse} | Extent RMSE: {extent_train_rmse} \n Testing: \n SIC RMSE: {sic_test_rmse} | Extent RMSE: {extent_test_rmse}")

print(f"Training: \n SIC nRMSE: {sic_train_nrmse} | Extent nRMSE: {extent_train_nrmse} \n Testing: \n SIC nRMSE: {sic_test_nrmse} | Extent nRMSE: {extent_test_nrmse}")


# Post-Processing
with open("/home/ekim33/reu2021_team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
	land_mask = np.squeeze(np.load(f))

# Multiply predicted and real SIC by land mask. Threshold predicted SIC to [0, 100] range.
land_mask = land_mask[408:504]
pred_ice = np.squeeze(sic_test_pred)
real_ice = np.squeeze(y_test)
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
print("Post-Processed MSE: ", mse, "\n", "RMSE: ", rmse, "\n")

# Output files for further evaluation
with open("/home/ekim33/reu2021_team1/research/evaluation/pred_ice_multiout_cnn_post_1_32.npy", "wb") as f:
	np.save(f, pred_ice)
with open("/home/ekim33/reu2021_team1/research/evaluation/real_ice_multiout_cnn_post_1_32.npy", "wb") as f:
	np.save(f, real_ice)
with open("/home/ekim33/reu2021_team1/research/evaluation/pred_extent_multiout_cnn_lag_one.npy", "wb") as f:
	np.save(f, extent_test_pred)
with open("/home/ekim33/reu2021_team1/research/evaluation/real_extent_multiout_cnn_lag_one.npy", "wb") as f:
	np.save(f, y_extent_test)


