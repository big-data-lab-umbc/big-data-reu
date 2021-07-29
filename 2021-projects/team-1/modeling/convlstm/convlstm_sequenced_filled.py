import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.compat.v1.keras import backend as K
from sklearn.preprocessing import MinMaxScaler

'''
Author: Peter I. Kruse
This document contains code for a ConvLSTM neural network predicting SIC per pixel and per month for spatio-temporal image data.
'''

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_train_rolling_filled_final.npy", "rb") as f:
        X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_train_rolling_filled_final.npy", "rb") as f:
        y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/X_test_rolling_filled_final.npy", "rb") as f:
        X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/Filled Rolling Data/y_test_rolling_filled_final.npy", "rb") as f:
        y_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/modeling/y_land_mask_actual.npy", "rb") as f:
        y_land_mask = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/y_extent_train_rolling_final.npy", "rb") as f:
        y_extent_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/convlstm/y_extent_test_rolling_final.npy", "rb") as f:
        y_extent_test = np.load(f)

# reshape y_land_mask
y_land_mask = y_land_mask.reshape(448, 304, 1)

from sklearn.metrics import mean_squared_error
# define custom mse loss, which applies land mask to each output of the network.
def custom_mse(y_true, y_pred):
	y_pred_masked = tf.math.multiply(y_pred, y_land_mask)
	y_true_masked = tf.math.multiply(y_true, y_land_mask)
	squared_resids = tf.square(y_true_masked - y_pred_masked)
	loss = tf.reduce_mean(squared_resids)
	return loss

#load per-pixel area file
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/preprocessing/area_size.npy", "rb") as f:
        areas = np.load(f)

#calculate ice extent
def calc_ice_extent(array):
        array = array / 100.0
        # use simple extent calculation
        tot_ice_extent = np.sum(np.multiply(array > 15.0, areas), axis=(1,2)) / 1e6
        return tot_ice_extent

# define ConvLSTM model
def create_convLSTM_image():
	#add ConvLSTM layers
	inputs = keras.layers.Input(shape=X_train.shape[1:])
	x = keras.layers.ConvLSTM2D(16, (5,5), padding="same", input_shape = (12, 448, 304, 11), return_sequences=False,
		activation="relu", data_format = 'channels_last')(inputs)
	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
	x = keras.layers.Conv2D(128, (5,5), padding="same", activation="relu")(x)
	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
	x = keras.layers.Conv2D(32, (5,5), padding="same", activation="relu")(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(256, activation="relu")(x)
	x = keras.layers.Dense(512, activation="relu")(x)
	x = keras.layers.Dense(448*304, activation="linear")(x)
	sic_output = keras.layers.Reshape((448, 304, 1), input_shape = (448*304,))(x)

        #initialize model
	model = keras.models.Model(inputs=inputs,
		outputs=sic_output,
		name="SIC_net")
        
	#compile model
	model.compile(optimizer="adamax", loss=custom_mse, metrics=[keras.metrics.RootMeanSquaredError()])
	return model

# define loss weights for each output
sample_weight = np.ones(shape=(len(y_train),))
train_extent = calc_ice_extent(y_train)
sample_weight[9::12]=1.2
print("first 5 sample weights: {}".format(sample_weight[0:5]))

# define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

# call model creation
convLSTM_image = create_convLSTM_image()

# print model summary
print(convLSTM_image.summary())

# fit model
history2 = convLSTM_image.fit(x=X_train, y=y_train,
	batch_size=4,
	epochs=1000,
	validation_split = .2,
	sample_weight=sample_weight,
	callbacks=[early_stopping])

# save model
convLSTM_image.save("convLSTM_image")

# image output
# train prediction
image_train_preds = convLSTM_image.predict(X_train)
image_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), image_train_preds.flatten()))

print("Image Concentration Train RMSE: {}".format(image_train_rmse))
print("Image Concentration Train NRMSE: {}".format(image_train_rmse / np.mean(y_train)))
print("Image Concentration Train NRMSE (std. dev.): {}".format(image_train_rmse / np.std(y_train)))
print("Train Prediction Shape: {}".format(image_train_preds.shape))

# test prediction
image_test_preds = convLSTM_image.predict(X_test)
image_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), image_test_preds.flatten()))

print("Image Concentration Test RMSE: {}".format(image_test_rmse))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.mean(y_test)))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.std(y_test)))
print("Test Prediction Shape: {}".format(image_test_preds.shape))

# save image outputs:
print(image_test_preds.shape)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/convlstm_image_rolling_filled_preds.npy", "wb") as f:
  np.save(f, image_test_preds)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/convlstm_image_rolling_filled_actual.npy", "wb") as f:
  np.save(f, y_test)

# calculate predicted extent
# train
train_pred_extent = calc_ice_extent(image_train_preds)
train_actual_extent = calc_ice_extent(y_train)
train_extent_rmse = math.sqrt(mean_squared_error(train_actual_extent, train_pred_extent))
print("Last Month Predicted Extent(Train): {}".format(train_pred_extent[-1]))
print("Last Month Actual Extent (Train): {}".format(train_actual_extent[-1]))
print("Train Extent RMSE: {}".format(train_extent_rmse))
print("Train Extent NRMSE: {}".format(train_extent_rmse / np.mean(train_actual_extent)))
print("Train Extent NRMSE (std. dev.): {}".format(train_extent_rmse / np.std(train_actual_extent)))

# test
test_pred_extent = calc_ice_extent(image_test_preds)
test_actual_extent = calc_ice_extent(y_test)
test_extent_rmse = math.sqrt(mean_squared_error(test_actual_extent, test_pred_extent))
print("Last Month Predicted Extent(Test): {}".format(test_pred_extent[-1]))
print("Last Month Actual Extent (Test): {}".format(test_actual_extent[-1]))
print("Test Extent RMSE: {}".format(test_extent_rmse))
print("Test Extent NRMSE: {}".format(test_extent_rmse / np.mean(test_actual_extent)))
print("Test Extent NRMSE (std. dev.): {}".format(test_extent_rmse / np.std(test_actual_extent)))

# save extent outputs:
print(image_test_preds.shape)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/convlstm_extent_rolling_filled_preds.npy", "wb") as f:
  np.save(f, test_pred_extent)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/GitHub/evaluation/convlstm/convlstm_extent_rolling_filled_actual.npy", "wb") as f:
  np.save(f, test_actual_extent)


import matplotlib.pyplot as plt

# Plot Loss
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss (Sea Ice Extent)')
plt.xlabel('Epoch')
plt.ylabel('Masked MSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Rolling_Filled_Extent_ConvLSTM_Loss_Plot.png')

# Plot Predicted vs. Actual Values (Sea Ice Extent)
# train
fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(train_actual_extent.shape[0])], train_pred_extent, c='b', label='Predicted')
ax.plot([i+1 for i in range(train_actual_extent.shape[0])], train_actual_extent, c='r', label='Actual')
ax.set_title('Sea Ice Extent by Month (Training Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('Rolling_Filled_Extent_ConvLSTM_train_pred_vs_actual.png')

# test
fig = plt.figure(figsize = (24, 6))
ax = fig.add_subplot(111)
ax.plot([i+1 for i in range(test_actual_extent.shape[0])], test_pred_extent, c='b', label='Predicted Extent')
ax.plot([i+1 for i in range(test_actual_extent.shape[0])], test_actual_extent, c='r', label='Actual Extent')
ax.set_title('Sea Ice Extent by Month (Testing Data)')
ax.set_xlabel('Month')
ax.set_ylabel('Sea Ice Extent (in $km^2$)')
ax.legend()
ax.grid(True)
fig.savefig('Rolling_Filled_Extent_ConvLSTM_test_pred_vs_actual.png')
