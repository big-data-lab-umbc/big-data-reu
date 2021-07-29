'''
Purpose: Build and train a CNN model which outputs a 448 by 304 grid of predicted sea ice concentrations for each month.

Data: One-month lag and filled North Pole Hole
Train: 1979-2012
Test: 2013-2020
Loss Function: Land Mask and SIE loss
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
	real_extent = np.load(f)

with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/area_size.npy", "rb") as f:
	areas = np.load(f)

X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)
real_extent_train = real_extent[1:408]
real_extent_test = real_extent[408:504]

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

def custom_rmse(y_true, y_pred):
	y_true = np.squeeze(y_true[:, :, :, :-1])
	#y_pred = y_pred[:, :, :, :-1]
	ext_arr = y_true[:, :, : -1]	
	extents = tf.reduce_mean(ext_arr, axis=(1, 2))	

	# Ice concentration Loss
	y_pred = tf.math.multiply(y_pred, y_land_mask)
	y_true = tf.math.multiply(y_true, y_land_mask)
	squared_difference = tf.square(y_true - y_pred)
	means = tf.reduce_mean(squared_difference)

	# Ice extent Loss
	y_bits = tf.cast(y_pred > 15, tf.float32)
	pred_extent = tf.math.reduce_sum(tf.math.multiply(y_bits, areas), axis=(1,2)) / 1e6
	extent_means = tf.reduce_mean(tf.square(pred_extent - extents))
	return tf.add(means, extent_means)

def rmse_metric(y_true, y_pred):
	y_true = np.squeeze(y_true[:, :, :, :-1])
	#y_pred = y_pred[:, :, :, :-1]
	return tf.reduce_mean(tf.square(y_true - y_pred))

def data_generator(data, targets, extents, batch_size):
	batches = (len(data) + batch_size - 1) // batch_size
	while(True):
		for i in range(batches):
			data_batch = data[i*batch_size : (i+1)*batch_size]
			targets_batch = targets[i*batch_size : (i+1)*batch_size]
			extents = targets[i*batch_size : (i + 1)*batch_size]	
			yield [data_batch, targets_batch, extents]

def create_cnn():
	inputs = keras.layers.Input(shape=X_train.shape[1:])
	x = keras.layers.Conv2D(128, (5,5), input_shape = (448, 304, 11))(inputs)
	x = keras.layers.MaxPooling2D((2,2))(x)
	x = keras.layers.Conv2D(32, (5,5), activation="relu")(x)
	x = keras.layers.MaxPooling2D((2,2))(x)
	x = keras.layers.Conv2D(8, (5,5), activation="relu")(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(256, activation="relu")(x)
	x = keras.layers.Dense(448*304, activation="linear")(x)
	y_pred = keras.layers.Reshape((448, 304), input_shape = (448*304,))(x)
	
	cnn = keras.models.Model(inputs=inputs, outputs=y_pred)
	cnn.compile(optimizer="adamax", loss=custom_rmse, metrics=rmse_metric, run_eagerly=True)
	return cnn

cnn = create_cnn()
print(cnn.summary())
real_extent_train_rep = np.repeat(real_extent_train, 448*304, axis=0).reshape(real_extent_train.shape[0], 448, 304)
y_train_ext = np.stack((y_train, real_extent_train_rep), axis=-1)
real_extent_test_rep = np.repeat(real_extent_test, 448*304, axis=0).reshape(real_extent_test.shape[0], 448, 304)
y_test_ext = np.stack((y_test, real_extent_test_rep), axis=-1)

history = cnn.fit(X_train, y_train_ext,
		batch_size=4,
		epochs=400,
		validation_data=(X_test, y_test_ext),
		callbacks=keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True))

train_loss, train_mse = cnn.evaluate(X_train, y_train_ext)
print("Train rMSE: {:.4f}\nTest Loss: {:.4f}".format(math.sqrt(train_mse), train_loss))

test_loss, test_mse = cnn.evaluate(X_test, y_test_ext)
print("Test rMSE: {:.4f}\nTest Loss: {:.4f}".format(math.sqrt(test_mse), test_loss))

pred_ice = np.squeeze(cnn.predict(X_test))
print(pred_ice.shape)

with open("/home/ekim33/reu2021_team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
	land_mask = np.load(f)

land_mask = land_mask[408:-6]
real_ice = y_test
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

mse = mean_squared_error(real_ice.flatten(), pred_ice.flatten())
rmse = math.sqrt(mse)
print("Post-Processed MSE: ", mse, "\n", "RMSE: ", rmse, "\n")

with open("/home/ekim33/reu2021_team1/research/evaluation/pred_ice_extent_maskthresh_lag_one_small_batch.npy", "wb") as f:
	np.save(f, pred_ice)
with open("/home/ekim33/reu2021_team1/research/evaluation/real_ice_extent_maskthresh_lag_one_small_batch.npy", "wb") as f:
	np.save(f, real_ice)
