import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.compat.v1.keras import backend as K
from sklearn.preprocessing import MinMaxScaler

'''
This document contains code for a ConvLSTM neural network predicting SIC per pixel and per month for spatio-temporal image data.
'''

# Data loading
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_train_rolling_filled_final.npy", "rb") as f:
        X_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_train_rolling_filled_final.npy", "rb") as f:
        y_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/X_test_rolling_filled_final.npy", "rb") as f:
        X_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_test_rolling_filled_final.npy", "rb") as f:
        y_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_actual.npy", "rb") as f:
        y_land_mask = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_train_rolling_final.npy", "rb") as f:
        y_extent_train = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/slurm/y_extent_test_rolling_final.npy", "rb") as f:
        y_extent_test = np.load(f)
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/plotting/real_ice_extents.npy", "rb") as f:
        y_extent = np.load(f)

def custom_activation(inp):
        return tf.clip_by_value(tf.multiply(inp, y_land_mask), clip_value_min = 0.0, clip_value_max = 100.0)

'''
# standardize data
X_train_norm = (X_train - X_train.mean())/X_train.std()
y_train_norm = (y_train - y_train.mean())/y_train.std()
X_test_norm = (X_test - X_test.mean())/X_test.std()
y_test_norm = (y_test - y_test.mean())/y_test.std()
'''

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
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/area_size.npy", "rb") as f:
        areas = np.load(f)

#calculate ice extent
def calc_ice_extent(array):
        array = array / 100.0
        print("array.shape:", array.shape)
        print("areas.shape:", areas.shape)
        # use simple extent calculation
        tot_ice_extent = np.sum(np.multiply(np.squeeze(array) > 15.0, areas), axis=(1,2)) / 1e6
        return tot_ice_extent

# define ConvLSTM model
def create_convLSTM_image():
	#add ConvLSTM layers
	inputs = keras.layers.Input(shape=X_train.shape[1:])
	x = keras.layers.ConvLSTM2D(16, (5,5), padding="same", return_sequences=False,
		activation="relu", data_format = 'channels_last')(inputs)
	#x = keras.layers.MaxPooling3D((2,2,2), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	#x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=True, activation="relu", data_format = 'channels_last')(x)
	#x = keras.layers.MaxPooling3D((2,2,2), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	#x = keras.layers.ConvLSTM2D(8, (5,5), padding="same", return_sequences=False, activation="relu", data_format = 'channels_last')(x)
	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv2D(128, (5,5), padding="same", activation="relu")(x)
	#x = keras.layers.MaxPooling2D((4,4), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	#x = keras.layers.Conv2D(128, (5,5), padding="same", activation="relu")(x)
	x = keras.layers.MaxPooling2D((2,2), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Conv2D(32, (5,5), padding="same", activation="relu")(x)
	#x = keras.layers.MaxPooling2D((4,4), padding='same')(x)
	#x = keras.layers.BatchNormalization()(x)
	#x = keras.layers.Conv2D(32, (5,5), padding="same", activation="relu")(x)
	x = keras.layers.Flatten()(x)
	#x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(256, activation="relu")(x)
	#x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(512, activation="relu")(x)
	#x = keras.layers.BatchNormalization()(x)
	x = keras.layers.Dense(448*304, activation="linear")(x)
	sic_output = keras.layers.Reshape((448, 304, 1), input_shape = (448*304,))(x)

        #input_mask = keras.layers.Input(shape=y_train_mask.shape[1:])
        #loss_inp = keras.layers.Input(shape=y_train.shape[1:])
	model = keras.models.Model(inputs=inputs,
		outputs=sic_output,
		name="SIC_net")
        #compile model
	model.compile(optimizer="adamax", loss=custom_mse, metrics=[keras.metrics.RootMeanSquaredError()])
	return model

sample_weight = np.ones(shape=(len(y_train),))
train_extent = calc_ice_extent(y_train)

#for i in range(len(sample_weight)):
#	sample_weight[i] = np.abs((train_extent[i] - np.mean(train_extent)) / np.mean(train_extent))

sample_weight[9::12]=1.2

print("first 5 sample weights: {}".format(sample_weight[0:5]))

'''
def exp_decay(epoch, lr):
	#init_rate = 0.1
	k = 0.1
	learning_rate = lr * math.exp(-k*epoch)
	return learning_rate
learning_rate = keras.callbacks.LearningRateScheduler(exp_decay)
'''

early_stopping = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

convLSTM_image = create_convLSTM_image()
print(convLSTM_image.summary())
history2 = convLSTM_image.fit(x=X_train, y=y_train,
	batch_size=4,
	epochs=50,
	validation_split = .2,
	#sample_weight=sample_weight,
	callbacks=[early_stopping])
convLSTM_image.save("convLSTM_image")

# image output
image_train_preds = convLSTM_image.predict(X_train, batch_size=4)
print("image_train_preds:", image_train_preds.flatten())
print("y_train:", y_train.flatten())
#image_train_mse, image_train_rmse = convLSTM_image.evaluate(X_train, y_train)

#image_train_preds = (image_train_preds_norm * y_train.std()) + y_train.mean()
image_train_rmse = math.sqrt(mean_squared_error(y_train.flatten(), image_train_preds.flatten()))

print("Image Concentration Train RMSE: {}".format(image_train_rmse))
print("Image Concentration Train NRMSE: {}".format(image_train_rmse / np.mean(y_train)))
print("Image Concentration Train NRMSE (std. dev.): {}".format(image_train_rmse / np.std(y_train)))
print("Train Prediction Shape: {}".format(image_train_preds.shape))

image_test_preds = convLSTM_image.predict(X_test, batch_size=4)
#image_test_mse, image_test_rmse = convLSTM_image.evaluate(X_test, y_test)

#image_test_preds = (image_test_preds_norm * y_test.std()) + y_test.mean()
image_test_rmse = math.sqrt(mean_squared_error(y_test.flatten(), image_test_preds.flatten()))

print("Image Concentration Test RMSE: {}".format(image_test_rmse))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.mean(y_test)))
print("Image Concentration Test NRMSE: {}".format(image_test_rmse / np.std(y_test)))
print("Test Prediction Shape: {}".format(image_test_preds.shape))

# save image outputs:
print(image_test_preds.shape)
with open("convlstm_image_rolling_filled_preds.npy", "wb") as f:
  np.save(f, image_test_preds)
with open("convlstm_image_rolling_filled_actual.npy", "wb") as f:
  np.save(f, y_test)

# Post-Processing
with open("/umbc/xfs1/cybertrn/reu2021/team1/research/preprocessing/y_land_mask_final_whole.npy", "rb") as f:
        land_mask = np.squeeze(np.load(f))

land_mask = land_mask[420:504]
pred_ice = np.squeeze(image_test_preds)
real_ice = np.squeeze(y_test)
pred_ice = tf.clip_by_value(np.multiply(pred_ice, land_mask), clip_value_min = 0.0, clip_value_max = 100.0).numpy()
real_ice = np.multiply(real_ice, land_mask)

# save post-processed output
with open("postproc_convlstm_image_rolling_preds.npy", "wb") as f:
        np.save(f, pred_ice)
with open("postproc_convlstm_image_rolling_actual.npy", "wb") as f:
        np.save(f, real_ice)

#calculate predicted extent
train_pred_extent = calc_ice_extent(image_train_preds)
train_actual_extent = calc_ice_extent(y_train)
train_extent_rmse = math.sqrt(mean_squared_error(train_actual_extent, train_pred_extent))
print("Last Month Predicted Extent(Train): {}".format(train_pred_extent[-1]))
print("Last Month Actual Extent (Train): {}".format(train_actual_extent[-1]))
print("Train Extent RMSE: {}".format(train_extent_rmse))
print("Train Extent NRMSE: {}".format(train_extent_rmse / np.mean(train_actual_extent)))
print("Train Extent NRMSE (std. dev.): {}".format(train_extent_rmse / np.std(train_actual_extent)))

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
with open("convlstm_extent_rolling_filled_preds.npy", "wb") as f:
  np.save(f, test_pred_extent)
with open("convlstm_extent_rolling_filled_actual.npy", "wb") as f:
  np.save(f, test_actual_extent)


import matplotlib.pyplot as plt
# Plot Loss (Sea Ice Extent)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model Loss (Sea Ice Extent)')
plt.xlabel('Epoch')
plt.ylabel('Masked MSE')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Rolling_Filled_Extent_ConvLSTM_Loss_Plot.png')


# Plot Predicted vs. Actual Values (Sea Ice Extent)
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
